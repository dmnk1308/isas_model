# load utils
from models import *
from helpers import *
from mesh_render import *

# load modules
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from unet import pretrain_unet
import wandb
import pandas as pd


def training_iter(model, dl_list, optimizer, criterion, device = None, img_list = None, verbose = True, batch_size = None, 
                sample_size = None, lungs_per_batch = 3, spatial_feat = False, xy_feat = False, keep_spatial=False, 
                point_resolution = 64, no_encoder = False, train_lungs = []):
    model.train()
    pred = []
    true = []
    loss_epoch = []
    kl_list = []
    mse_list = []


    lungs = np.arange(len(dl_list))         
    aug_multi = int(len(lungs)/len(train_lungs))
    train_lungs = aug_multi*train_lungs
    
    # create vector with lung ids
    iterator = [iter(dl) for dl in dl_list]         # create an iterator for each lung dataloader (shuffles coordinates randomly)
    
        # 1. wir mischen die Lungen durch -> temporäre Lungen
        # 2. wir nehmen uns die ersten lungs_per_batch Lungen
        # 3. wir rufen eine Iteration des Iterators auf und lassen das Modell rechnen
        #     Falls Fehler: überpringe Lunge und Lösche Lunge aus temporären Lungen
        # 4. wir löschen die Lunge aus dem temporären Lungen
        # -> 1. bis alle Iterationen durch sind

    # training iterations 
    for global_count in tqdm(range(int(sample_size/batch_size)), disable = not verbose):   # compute number of iterations in 1 epoch (sample_size/batch_size)

        np.random.shuffle(lungs)                        # shuffle lung ids randomly
        lungs_tmp = lungs
        train_lungs_tmp = [train_lungs[i] for i in lungs]

        # devide lung batches
        for lung_iter in range(int(np.ceil(len(lungs)/lungs_per_batch))):

            if len(lungs_tmp) == 0:     # if no lungs are left skip the loop
                continue

            y_hat_list = []
            t_list = []
            mu_list = []
            logsigma_list = []
            counter = 0

            while counter <= lungs_per_batch and len(lungs_tmp) > 0:                                 # for each "random" lung train model
                
                i = lungs_tmp[0]
                lung = lungs_tmp[0]

                img = img_list[i]
                slices_max = img.shape[0]
                # randomly leave out slices
                # 1. decide whether to leave slices out or not

                leave_out = random.randint(0,1)
                if leave_out == 1:# and spatial_feat == False:
                    # determine how many slices are left out
                    number_leave_out = random.randint(10, img.shape[0]-10)
                    slices_leave_out = np.random.randint(0, img.shape[0], number_leave_out).tolist()
                    slices_leave_out.sort()
                    img = img[slices_leave_out]
                    slices_leave_out = torch.tensor(slices_leave_out)
                else:
                    slices_leave_out = torch.arange(slices_max)   

                img = img.unsqueeze(1)

                try:
                    x, filter_in ,t,_ = next(iterator[i])     # get coordinates from lung iterator
                except:                                        # if no iteration is left, delete lung from lungs_tmp
                    lungs_tmp = lungs_tmp[1:]
                    train_lungs_tmp = train_lungs_tmp[1:]
                    break

                # predict
                t = t.float().to(device)
                t = torch.unsqueeze(t, -1)
                
                if spatial_feat == True or xy_feat == True:
                    # with profile(activities=[
                    #         ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                    #     with record_function("model_inference"):
                    y_hat = model([x.float().to(device).detach(), img.float().to(device), filter_in.to(device)],  slice_index = slices_leave_out, slice_max = slices_max, resolution = point_resolution, z_resolution = slices_max)
                    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

                # elif vae == True:
                #     y_hat, mu, logsigma = model([x.float().to(device), img.float().to(device)])   
                #     mu_list.append(mu)
                #     logsigma_list.append(logsigma)

                elif no_encoder == True:
                    y_hat = model([x.float().to(device), torch.tensor(lung).to(device)])

                else:
                    y_hat = model([x.float().to(device), img.float().to(device)], slice_index = slices_leave_out, slice_max = slices_max)

                y_hat_list.append(y_hat)
                t_list.append(t)

                pred.append(np.squeeze(np.round(torch.sigmoid(y_hat).detach().cpu().numpy())))
                true.append(np.squeeze(t.detach().cpu().numpy()))

                lungs_tmp = lungs_tmp[1:]
                train_lungs_tmp = train_lungs_tmp[1:]

                counter += 1

            if len(y_hat_list) == 0:
                continue

            y_hat = torch.cat(y_hat_list)
            t = torch.cat(t_list)

            # if vae == True:
            #     mu = torch.cat(mu_list)
            #     logsigma = torch.cat(logsigma_list)
            #     loss, mse_loss, kl_loss = criterion(y_hat, t, mu, logsigma)
            #     kl_list.append(kl_loss.detach().cpu().numpy())
            #     mse_list.append(mse_loss.detach().cpu().numpy())

            # else:
            loss = criterion(y_hat, t)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_epoch.append(loss.detach().cpu().numpy())  

    pred = np.concatenate(pred)
    true = np.concatenate(true)
    loss_epoch = np.mean(np.array(loss_epoch))

    acc_epoch = sum(pred == true)/len(true)
    iou_epoch = iou(torch.from_numpy(np.expand_dims(pred,-1)), torch.from_numpy(np.expand_dims(true,-1)))
    dice_epoch = dice_coef(torch.from_numpy(np.expand_dims(pred,-1)), torch.from_numpy(np.expand_dims(true,-1)))    

    # if vae == True:
    #     kl_epoch = np.mean(np.array(kl_list))
    #     mse_epoch = np.mean(np.array(mse_list))
    #     print("KL: ", kl_epoch)
    #     print("MSE: ", mse_epoch)

    return model, loss_epoch, acc_epoch, iou_epoch, dice_epoch

def validation(model, dl_list, criterion, device = None, img_list = None, spatial_feat = False, 
    resolution = 128, xy_feat = False, keep_spatial = False, no_encoder = False, val_lungs = []):

    # validation on last lung loaded
    model.eval()
    pred = []
    true = []
    loss_list = []
    mu_list = []
    logsigma_list = []

    for dataloader, img, lung in zip(dl_list,img_list, val_lungs):
        img = img.unsqueeze(1)
        for x,filter_in,t,_ in dataloader:
            t = t.float().to(device)
            t = torch.unsqueeze(t, -1)
            
            if spatial_feat == True or xy_feat == True:
                y_hat = model([x.float().to(device), img.float().to(device), filter_in.to(device)],resolution = resolution, z_resolution = img.shape[0])  
            
            # elif vae == True:
            #     y_hat, mu, logsigma = model([x.float().to(device), img.float().to(device)])   
            #     mu_list.append(mu.detach())
            #     logsigma_list.append(logsigma.detach())

            elif no_encoder == True:
                y_hat = model([x.float().to(device), torch.tensor(lung).to(device)])
            
            else:
                y_hat = model([x.float().to(device), img.float().to(device)])

            pred.append(np.squeeze(np.round(torch.sigmoid(y_hat).detach().cpu().numpy())))
            true.append(np.squeeze(t.detach().cpu().numpy()))

            # if vae == True:
            #     mu = torch.cat(mu_list)
            #     logsigma = torch.cat(logsigma_list)
            #     loss, mse_loss, kl_loss = criterion(y_hat, t, mu, logsigma)

            #else:
            loss = criterion(y_hat, t)

            loss_list.append(loss.detach().cpu().numpy())
    pred = np.concatenate(pred)
    true = np.concatenate(true)

    # compute metrics
    acc_val = np.sum(pred == true)/len(true)
    loss_val = np.mean(np.array(loss_list))
    iou_val = iou(torch.from_numpy(np.expand_dims(pred,-1)), torch.from_numpy(np.expand_dims(true,-1))).numpy()
    dice_val = dice_coef(torch.from_numpy(np.expand_dims(pred,-1)), torch.from_numpy(np.expand_dims(true,-1))).numpy()

    return loss_val, acc_val, iou_val, dice_val


def train(model, wandb, model_name, num_lungs, lr, epochs, batch_size, patience, point_resolution, 
        img_resolution, shape_resolution = 128, verbose = True, device=None, sample_size=None, val_lungs = [], test_lungs =[],
        weight_decay = 0., augmentations = True, spatial_feat = False, xy_feat = False, keep_spatial = False, global_local = False, 
        num_blocks = 5, num_feat = 32, no_encoder = False, visualize_epochs = True, **_):

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

    if spatial_feat == True or global_local == True:
        model = pretrain_unet(model, wandb = None, lungs = train_lungs, val_lungs = val_lungs, feat = num_feat, num_blocks = num_blocks, epochs = 10 , retrain = True, verbose = True, device = device, resolution = img_resolution)
        
    # load data
    data_list_train, img_list_train, mask_sample_list_train = load_data(train_lungs, 
        train = True,
        point_resolution = point_resolution, 
        img_resolution = img_resolution, 
        batch_size = batch_size, 
        sample_size = sample_size, 
        augmentations=augmentations)

    data_list_val, img_list_val, mask_sample_list_val = load_data(val_lungs, 
        train = False,
        point_resolution = shape_resolution, 
        img_resolution = img_resolution,
        augmentations=False
        )

    # train with binary cross entropy and ADAM optimizer
    # if vae == True:
    #     criterion = vae_loss
    # else:
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = patience)

    # metric lists
    loss = []
    acc = []
    iou_values = []
    dice = []

    # compute mean and standard deviation of images
    model.img_mean = nn.parameter.Parameter(torch.mean(torch.cat(img_list_train)), requires_grad = False)  
    model.img_std = nn.parameter.Parameter(torch.std(torch.cat(img_list_train)), requires_grad = False)  

    # standardize images    
    img_list_train = [(img-model.img_mean)/model.img_std for img in img_list_train] 
    img_list_val = [(img-model.img_mean)/model.img_std for img in img_list_val] 

    ref_iou = 0

    # DATALOADER VALIDATION
    dl_list_val = []
    for d,m in zip(data_list_val,mask_sample_list_val):
        dl_list_val.append(DataLoader(d, batch_size = 100000, shuffle = False))

    for epoch in tqdm(range(epochs), disable = not verbose):
        
        # EPOCH SPECIFIC DATALOADER TRAIN
        dl_list_train = []
        for d,m in zip(data_list_train,mask_sample_list_train):
            sample_mask = voxel_sample(m, flatten = False, n_samples = sample_size, verbose = True)
            sample_mask =  np.array(np.where(sample_mask.reshape(-1))).squeeze()
            d.filter_mask = sample_mask
            dl_list_train.append(DataLoader(d, batch_size = batch_size, shuffle = True))

        # train model
        model, loss_epoch, acc_epoch, iou_epoch, dice_epoch = training_iter(model = model, dl_list = dl_list_train, 
            optimizer = optimizer, criterion = criterion, device = device, img_list=img_list_train, verbose = verbose, 
            batch_size=batch_size, sample_size=sample_size, spatial_feat = spatial_feat, xy_feat = xy_feat, keep_spatial=keep_spatial,
            point_resolution=point_resolution, no_encoder = no_encoder, train_lungs = train_lungs)

        # validate model
        loss_val, acc_val, iou_val, dice_val = validation(model = model, dl_list = dl_list_val, criterion = criterion, device = device, 
            img_list = img_list_val, spatial_feat = spatial_feat, resolution = shape_resolution, xy_feat = xy_feat, keep_spatial = keep_spatial,
            no_encoder = no_encoder, val_lungs = val_lungs)

        try:
            wandb.log({"training_loss": loss_epoch,
                        "training_acc": acc_epoch,
                        "training_iou": iou_epoch,
                        "val_loss": loss_val,
                        "val_acc": acc_val,
                        "val_iou": iou_val})
        except:
            None

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
        scheduler.step(loss_epoch)

        # console output
        if verbose == True:
            print("\n####### Epoch ", str(epoch)," #######")
            print("\n## Training ##")
            print("Accuracy: ", np.round(acc_epoch, 5))
            print("IoU: ", np.round(iou_epoch.numpy(), 5)) 
            print("Dice: ", np.round(dice_epoch.numpy(),5)) 
            print("Loss: ", np.round(loss_epoch,5))  
            print("\n## Validation ##")
            print("Accuracy: ", np.round(acc_val, 5))
            print("IoU Validation: ", np.round(iou_val,5))
            print("Dice Validation: ", np.round(dice_val,5))
            print("Loss Validation: ", np.round(loss_val,5))    
        
        if visualize_epochs == True:
            visualize(model, wandb, np.array([0,1,2,3]), img_resolution=img_resolution, 
                shape_resolution = shape_resolution, device = device, model_name = str(model_name)+"_epoch_"+str(epoch), spatial_feat = spatial_feat, 
                xy_feat = xy_feat, latent_dim = 64, keep_spatial = keep_spatial, no_encoder = no_encoder, max_batch=100000)

    #torch.save(model.state_dict(), path)
    wandb.save(path,policy = "now")

    return model, acc, iou_values, dice, ref_iou

def visualize(model, wandb, lungs, img_resolution = 128, shape_resolution = 128, device = None, 
    model_name = "", spatial_feat = False, xy_feat = False, latent_dim = 64, keep_spatial = False, no_encoder = False, 
    max_batch = 64**3, **_):
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
        pred = model_to_voxel(model,device=device, img = img, resolution = mask.shape[1], z_resolution= mask.shape[-1], max_batch = max_batch, spatial_feat = spatial_feat, 
            xy_feat = xy_feat, latent_dim = latent_dim, keep_spatial = keep_spatial, no_encoder=no_encoder, lung = i)   
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
    
    results.to_csv("results/tmp_val_metrics.csv")
    try:
        wandb.save("results/"+str(model_name) +"_tmp_val_metrics.csv")
    except:
        None

def visualize_mask(start_lung, stop_lung, resolution = 128, device=None, **_):

    for i in tqdm(range(start_lung,stop_lung)):
        mask_raw = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()
        mask = np.where(mask_raw < 100,  0, 1)
        get_ply(mask = mask, ply_filename = "self_supervised_model/masks/lung_mask_"+str(i), from_mask = True, resolution = resolution, device = device)
