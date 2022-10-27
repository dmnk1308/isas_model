# load utils
from models import *
from helpers import *
from render import *

# load modules
import numpy as np
import torch
from tqdm import tqdm
import wandb
import pandas as pd
import random
import matplotlib.pyplot as plt

def training_iter(model, dl_list, optimizer, criterion, device = None, img_list = None, verbose = True, batch_size = None, 
                sample_size = None, lungs_per_batch = 3, no_encoder = False, train_lungs = []):
    model.train()
    loss_epoch = []
    iou_epoch = []
    dice_epoch = []
    acc_epoch = []
    
    # make list 
    lungs = np.arange(len(dl_list))         
    aug_multi = int(len(lungs)/len(train_lungs))
    train_lungs = np.array(aug_multi*train_lungs)
    
    # create iterator for each lung which shuffles coordinates randomly
    iterator = [iter(dl) for dl in dl_list]        
    
    # training iterations 
    for global_count in tqdm(range(int(sample_size/batch_size)), disable = not verbose):    # compute number of point iterations in 1 epoch (sample_size/batch_size) for each lung 
        np.random.shuffle(lungs)                                                            # shuffle lungs randomly
        lungs_tmp = lungs                                                                   # make temporary array for the CT iteration step
        train_lungs_tmp = train_lungs[lungs_tmp]
        
        # devide lung batches
        for lung_iter in range(int(np.ceil(len(lungs)/lungs_per_batch))):                   # computer number of CT iteration steps per point iteration

            if len(lungs_tmp) == 0:     # if no lungs are left skip the loop
                continue

            y_hat_list = []
            t_list = []
            counter = 0

            while counter <= lungs_per_batch and len(lungs_tmp) > 0:                        # for each "random" lung train model
                
                i = lungs_tmp[0]
                l = train_lungs_tmp[0]

                img = img_list[i]
                slices_max = img.shape[0]
                
                # randomly leave out slices
                # 1. decide whether to leave slices out or not
                leave_out = random.randint(0,1)
                
                if leave_out == 1:
                    # 2. determine how many slices to provide as input
                    number_input = random.randint(5, img.shape[0]-10)
                    slices_input = np.random.randint(0, img.shape[0], number_input).tolist()
                    slices_input.sort()
                    img = img[slices_input]
                    slices_input = torch.tensor(slices_input)
                else:
                    slices_input = torch.arange(slices_max)   

                img = img.unsqueeze(1)

                try:
                    x, filter_in ,t,_ = next(iterator[i])      # get data from lung iterator
                except:                                        # if no iteration is left, delete lung from lungs_tmp
                    lungs_tmp = lungs_tmp[1:]
                    train_lungs_tmp = train_lungs_tmp[1:]
                    break

                # predict
                t = t.float().to(device)
                t = torch.unsqueeze(t, -1)

                if no_encoder == True:
                    y_hat = model([x.float().to(device), torch.tensor(l).to(device)])

                else:
                    y_hat = model([x.float().to(device).detach(), img.float().to(device), None], slice_index = slices_input, slice_max = slices_max)
 
                y_hat_list.append(y_hat)
                t_list.append(t)

                lungs_tmp = lungs_tmp[1:]
                train_lungs_tmp = train_lungs_tmp[1:]

                counter += 1

            if len(y_hat_list) == 0:
                continue

            # prediction and true values for iteration step
            y_hat = torch.cat(y_hat_list)
            t = torch.cat(t_list)
            
            # compute BCE
            loss = criterion(y_hat, t)
            
            # backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # save metrics for iteration
            loss = loss.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu()
            t = t.detach().cpu()
            y_hat = torch.round(torch.sigmoid(y_hat))
            acc_epoch.append((torch.sum(y_hat == t)/len(t)).numpy())
            iou_epoch.append(iou(y_hat, t).numpy())
            dice_epoch.append(dice_coef(y_hat, t).numpy())
            loss_epoch.append(loss)  

    # compute mean metrics for epoch iterations
    loss_epoch = np.mean(np.array(loss_epoch))
    acc_epoch = np.mean(np.array(acc_epoch))
    iou_epoch = np.mean(np.array(iou_epoch))
    dice_epoch = np.mean(np.array(dice_epoch))
    
    return model, loss_epoch, acc_epoch, iou_epoch, dice_epoch

def validation(model, dl_list, criterion, device = None, img_list = None, no_encoder = False, val_lungs = []):
    model.eval()
    
    loss_list = []
    iou_list = []
    dice_list = []
    acc_list = []
    
    for dataloader, img, lung in zip(dl_list,img_list, val_lungs):
        img = img.unsqueeze(1)
        t_list = []
        y_hat_list = []
        
        for x,filter_in,t,_ in dataloader:
            t = t.float().to(device)
            t = torch.unsqueeze(t, -1)
            
            if no_encoder == True:
                y_hat = model([x.float().to(device), torch.tensor(lung).to(device)])
            
            else:
                y_hat = model([x.float().to(device), img.float().to(device), filter_in.to(device)])  

            t = t.detach().cpu()
            y_hat = y_hat.detach().cpu()
            t_list.append(t)
            y_hat_list.append(y_hat)

        t = torch.cat(t_list)        
        y_hat = torch.cat(y_hat_list)
                    
        loss = criterion(y_hat, t)    
        loss_list.append(loss.detach().cpu().numpy())
            
        y_hat = torch.round(torch.sigmoid(y_hat))
        
        acc_list.append((torch.sum(y_hat == t)/len(t)).numpy())
        iou_list.append(iou(y_hat, t).numpy())
        dice_list.append(dice_coef(y_hat, t).numpy())
            
    # compute mean metrics for validation lungs
    loss_val = np.mean(np.array(loss_list))
    acc_val = np.mean(np.array(acc_list))
    iou_val = np.mean(np.array(iou_list))
    dice_val = np.mean(np.array(dice_list))

    return loss_val, acc_val, iou_val, dice_val


def train(model, wandb, model_name, num_lungs, lr, epochs, batch_size, patience, point_resolution, 
        img_resolution, shape_resolution = 128, verbose = True, device=None, sample_size=None, val_lungs = [], test_lungs =[],
        weight_decay = 0., augmentations = True, spatial_feat = False, no_encoder = False, visualize_epochs = False, proportion = 1.0, border = True, random = False, unbalanced = True,
        batch_size_val = 2000, **_):

    # set seed
    torch.manual_seed(123)
    np.random.seed(123)

    # list with all lungs
    train_lungs = [i for i in range(num_lungs)]

    for i in val_lungs:
        try:
            train_lungs.remove(i)
        except:
            None

    for i in test_lungs:
        try:
            train_lungs.remove(i)
        except:
            None

    print("Training Lungs: ", train_lungs, "\nValidation Lungs: ", val_lungs)

    # load data
    dl_list_train, img_list_train = load_data(train_lungs, 
        train = True,
        point_resolution = point_resolution, 
        img_resolution = img_resolution, 
        batch_size = batch_size, 
        sample_size = sample_size, 
        augmentations=augmentations,
        proportion = proportion,
        border = border,
        random = random,
        unbalanced = unbalanced)

    dl_list_val, img_list_val = load_data(val_lungs, 
        train = False,
        point_resolution = shape_resolution, 
        img_resolution = img_resolution,
        augmentations=False,
        proportion=proportion,
        batch_size= batch_size_val
        )

    # train with binary cross entropy and ADAM optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = patience)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 6, gamma=0.1, verbose = True)

    # metric lists
    loss = []
    acc = []
    iou_values = []
    dice = []

    # compute mean and standard deviation of images
    if no_encoder == False:
        model.img_mean = nn.parameter.Parameter(torch.mean(torch.cat(img_list_train)), requires_grad = False)  
        model.img_std = nn.parameter.Parameter(torch.std(torch.cat(img_list_train)), requires_grad = False)  

        # standardize images    
        img_list_train = [(img-model.img_mean)/model.img_std for img in img_list_train] 
        img_list_val = [(img-model.img_mean)/model.img_std for img in img_list_val] 

    ref_iou = 0.

    for epoch in tqdm(range(epochs), disable = not verbose):        
        # train model
        model, loss_epoch, acc_epoch, iou_epoch, dice_epoch = training_iter(model = model, dl_list = dl_list_train, 
            optimizer = optimizer, criterion = criterion, device = device, img_list=img_list_train, verbose = verbose, 
            batch_size=batch_size, sample_size=sample_size, no_encoder = no_encoder, train_lungs = train_lungs)

        # validate model
        if no_encoder == False:
            try:
                loss_val, acc_val, iou_val, dice_val = validation(model = model, dl_list = dl_list_val, criterion = criterion, device = device, 
                    img_list = img_list_val, no_encoder = no_encoder, val_lungs = val_lungs)
            except:
                print("Validation on cpu.")
                model.to("cpu")
                loss_val, acc_val, iou_val, dice_val = validation(model = model, dl_list = dl_list_val, criterion = criterion, device = "cpu", 
                    img_list = img_list_val, no_encoder = no_encoder, val_lungs = val_lungs)
                model.to(device)
            try:
                wandb.log({"training_loss": loss_epoch,
                            "training_acc": acc_epoch,
                            "training_iou": iou_epoch,
                            "val_loss": loss_val,
                            "val_acc": acc_val,
                            "val_iou": iou_val})
            except:
                None

        else:
            print("No Encoder. No Validation.")
            loss_val, acc_val, iou_val, dice_val = [0,0,0,0]

        path = "model_checkpoints/final_models/"+model_name+".pt"

        # save model if iou_train has increased
        if (iou_val > ref_iou) or (no_encoder == True):
            torch.save(model.state_dict(), path)
            print("Model saved.")
            ref_iou = iou_val
        else:
            None

        # save metrics
        loss.append(loss_epoch)
        acc.append(acc_epoch)
        iou_values.append(iou_epoch)
        dice.append(dice_epoch)

        # make scheduler step
        scheduler.step()

        # console output
        if verbose == True:
            print("\n####### Epoch ", str(epoch)," #######")
            print("\n## Training ##")
            print("Accuracy: ", np.round(acc_epoch, 5))
            print("IoU: ", np.round(iou_epoch, 5)) 
            print("Dice: ", np.round(dice_epoch,5)) 
            print("Loss: ", np.round(loss_epoch,5))  
            print("\n## Validation ##")
            print("Accuracy: ", np.round(acc_val, 5))
            print("IoU Validation: ", np.round(iou_val,5))
            print("Dice Validation: ", np.round(dice_val,5))
            print("Loss Validation: ", np.round(loss_val,5))    
        
        if visualize_epochs == True:
            try:
                visualize(model, wandb, np.array([0,1,178]), img_resolution=img_resolution, 
                    shape_resolution = shape_resolution, device = device, model_name = str(model_name)+"_epoch_"+str(epoch), no_encoder = no_encoder, max_batch=batch_size_val)
            except:
                print("Visualize on cpu.")
                model.to("cpu")
                try:
                    visualize(model, wandb, np.array([0,1,178]), img_resolution=img_resolution, 
                        shape_resolution = shape_resolution, device = "cpu", model_name = str(model_name)+"_epoch_"+str(epoch), no_encoder = no_encoder, max_batch= batch_size_val)
                    model.to(device)
                except:
                    print("No Visualization Possible.")
                    model.to(device)
                    
    # optimize test features for decoder only model
    if no_encoder == True:
        ref_val = 0
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        epochs = 30
        for param in model.dec.parameters():
            param.requires_grad = False
        model.train()
        
        dl_list_test, img_list_test = load_data(test_lungs, 
            train = True,
            point_resolution = point_resolution, 
            img_resolution = img_resolution, 
            batch_size = batch_size, 
            sample_size = sample_size, 
            augmentations=augmentations,
            proportion = proportion,
            border = border,
            random = random,
            unbalanced = unbalanced)
        
        for epoch in tqdm(range(epochs), disable = not verbose):  
            model, loss_epoch, acc_epoch, iou_epoch, dice_epoch = training_iter(model = model, dl_list = dl_list_test, 
                optimizer = optimizer, criterion = criterion, device = device, img_list=img_list_test, verbose = verbose, 
                batch_size=batch_size, sample_size=sample_size, no_encoder = no_encoder, train_lungs = test_lungs)
            print("\n####### TEST LUNGS #######")
            print("Accuracy: ", np.round(acc_epoch, 5))
            print("IoU: ", np.round(iou_epoch, 5)) 
            print("Dice: ", np.round(dice_epoch,5)) 
            print("Loss: ", np.round(loss_epoch,5))  
            if ref_iou < iou_epoch:
                torch.save(model.state_dict(), path)
                ref_iou = iou_epoch
    
    torch.save(model.state_dict(), path)

    return model, acc, iou_values, dice, ref_iou

def visualize(model, wandb, lungs, img_resolution = 128, shape_resolution = 128, device = None, model_name = "", no_encoder = False, max_batch = 5000, **_):
    model.eval()
    
    # load lungs to visualize
    _, img_list, mask_list = load_data(lungs, 
        train = False,
        point_resolution = shape_resolution,
        img_resolution = img_resolution,
        return_mask = True,
        augmentations=False
        )

    iou_list = []
    dice_list = []
    acc_list = []

    for i, img, mask in zip(lungs, img_list, mask_list):
        mask = mask.moveaxis(0,-1)
        #pred = model_to_voxel(model,device=device, img = img, resolution = shape_resolution, max_batch = 64 ** 3)    
        pred = model_to_voxel(model,device=device, img = img, resolution = mask.shape[1], z_resolution= mask.shape[-1], max_batch = max_batch, no_encoder=no_encoder, lung = i)   
        pred = pred.cpu().numpy()
        pred = np.moveaxis(pred,-1,0)
        get_ply(mask = pred, ply_filename = "dump/"+model_name+"_lung_"+str(i), from_mask = True, resolution = shape_resolution, device = device)
        try:
            wandb.save("visualization/ply_data/dump/"+model_name+"_lung_"+str(i)+".ply", base_path = "visualization/ply_data", policy = "now")
        except:
            None
        pred = torch.from_numpy(pred).moveaxis(0,-1)
        dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
        iou_value = iou(torch.round(torch.sigmoid(pred)), mask)
        acc = np.sum(pred.flatten() == mask.numpy().flatten())/len(pred.flatten())

        print("Dice Coeff.: ", dice.numpy())
        print("IoU: ",iou_value.numpy())

        acc_list.append(acc)
        dice_list.append(dice)
        iou_list.append(iou_value)
        
    results = pd.DataFrame({"Lung": lungs,
        "Accuracy" : acc_list,
        "Dice" : dice_list,
        "IoU" : iou_list})
    
    results.to_csv("results/"+str(model_name) +"_tmp_val_metrics.csv")
    try:
        wandb.save("results/"+str(model_name) +"_tmp_val_metrics.csv", policy = "now")
    except:
        None


def visualize_mask(start_lung, stop_lung, resolution = 128, out_equal = False, equal = 10, out_top = False, 
                   out_bottom = False, out_bottom_top = False, device="cuda:0", **_):
    resolution = int(512/resolution)

    if out_equal == True:
        for i in tqdm(range(start_lung,stop_lung)):
            mask_raw = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()
            mask_raw = resize(mask_raw, 41)
            mask = np.where(mask_raw < 100,  0, 1)
            f = np.arange(0,mask.shape[0],equal)
            print(f)
            f_out = np.arange(0, mask.shape[0])
            f_out = np.delete(f_out, f, axis = 0)
            
            mask_in = mask.copy()
            mask_out = mask.copy()
            
            mask_in[f_out,:,:] = -1
            mask_out[f,:,:] = -1
            
            get_ply(mask = mask_out, ply_filename = "dump/lung_mask_"+str(i)+"_out_equal", from_mask = True, resolution = resolution, device = device)
            get_ply(mask = mask_in, ply_filename = "dump/lung_mask_"+str(i)+"_in_equal", from_mask = True, resolution = resolution, device = device)


    if out_top == True:
        for i in tqdm(range(start_lung,stop_lung)):
            mask_raw = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()[:,::resolution,::resolution]
            mask_raw = resize(mask_raw, 48)
            mask = np.where(mask_raw < 100,  0, 1)
            f = np.arange(mask.shape[0]-20,mask.shape[0])
            f_out = np.arange(0, mask.shape[0])
            f_out = np.delete(f_out, f, axis = 0)
            
            mask_in = mask.copy()
            mask_out = mask.copy()
            
            mask_in[f_out,:,:] = -1
            mask_out[f,:,:] = -1
            
            get_ply(mask = mask_out, ply_filename = "dump/lung_mask_"+str(i)+"_out_top", from_mask = True, resolution = resolution, device = device)
            get_ply(mask = mask_in, ply_filename = "dump/lung_mask_"+str(i)+"_in_top", from_mask = True, resolution = resolution, device = device)
        
    if out_bottom == True:
        for i in tqdm(range(start_lung,stop_lung)):
            mask_raw = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()[:,::resolution,::resolution]
            mask_raw = resize(mask_raw, 48)
            mask = np.where(mask_raw < 100,  0, 1)
            f = np.arange(0,20)
            f_out = np.arange(0, mask.shape[0])
            f_out = np.delete(f_out, f, axis = 0)
            
            mask_in = mask.copy()
            mask_out = mask.copy()
            
            mask_in[f_out,:,:] = -1
            mask_out[f,:,:] = -1
            
            get_ply(mask = mask_out, ply_filename = "dump/lung_mask_"+str(i)+"_out_bottom", from_mask = True, resolution = resolution, device = device)
            get_ply(mask = mask_in, ply_filename = "dump/lung_mask_"+str(i)+"_in_bottom", from_mask = True, resolution = resolution, device = device)
        
    if out_bottom_top == True:
        for i in tqdm(range(start_lung,stop_lung)):
            mask_raw = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()[:,::resolution,::resolution]
            mask_raw = resize(mask_raw, 48)
            mask = np.where(mask_raw < 100,  0, 1)
            f = np.arange(int(mask.shape[0]/2)-10,int(mask.shape[0]/2)+10)
            f_out = np.arange(0, mask.shape[0])
            f_out = np.delete(f_out, f, axis = 0)
            
            mask_in = mask.copy()
            mask_out = mask.copy()
            
            mask_in[f_out,:,:] = -1
            mask_out[f,:,:] = -1
            
            get_ply(mask = mask_out, ply_filename = "dump/lung_mask_"+str(i)+"_out_middle", from_mask = True, resolution = resolution, device = device)
            get_ply(mask = mask_in, ply_filename = "dump/lung_mask_"+str(i)+"_in_middle", from_mask = True, resolution = resolution, device = device)
        
    else:
        for i in tqdm(range(start_lung,stop_lung)):
            mask_raw = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()[:,::resolution,::resolution]
            mask_raw = resize(mask_raw, 48)
            mask = np.where(mask_raw < 100,  -1, 1)
            
            get_ply(mask = mask, ply_filename = "dump/lung_mask_"+str(i), from_mask = True, resolution = resolution, device = device)