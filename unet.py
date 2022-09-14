from random import shuffle
import torch 
import numpy
from os.path import exists
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from helpers import *
from models import *
from render import *
import wandb

def training_step_unet(model, dl, optimizer, device = None):
    model.train()
    pred = []
    true = []
    loss_epoch = []
    
    for x,t in dl:
        optimizer.zero_grad()
        t = t.to(device)
        y_hat = model(x.float().to(device))
        y_hat = torch.sigmoid(y_hat)
        loss = dice_loss(y_hat, t)
        loss.backward()
        optimizer.step()
        
        pred.append(np.squeeze(np.round(y_hat.detach().cpu().numpy().flatten())))
        true.append(np.squeeze(t.detach().cpu().numpy().flatten()))
        loss_epoch.append(loss.detach().cpu().numpy()) 
    
    pred = np.concatenate(pred)
    true = np.concatenate(true)
    loss_epoch = np.mean(np.array(loss_epoch))
    acc_epoch = np.sum(pred == true)/len(true)
    iou_epoch = iou(torch.from_numpy(pred), torch.from_numpy(true))
    dice_epoch = dice_coef(torch.from_numpy(pred), torch.from_numpy(true))   

    return loss_epoch, acc_epoch, iou_epoch, dice_epoch
    
def validation_unet(model, images_val, masks_val, device = None):
    model.eval()
    y_hat_list = []
    t_list = []

    for image, mask in zip(images_val, masks_val):
        y_hat = model(image.float().to(device).unsqueeze(1))
        y_hat = torch.sigmoid(y_hat.detach().cpu()).squeeze()
        y_hat_list.append(y_hat.flatten())
        t_list.append(mask.flatten())

    y_hat = torch.round(torch.concat(y_hat_list))
    t = torch.concat(t_list)
    acc_val = torch.sum(y_hat == t)/len(t)        
    iou_val = iou(y_hat, t)
    dice_val = dice_coef(y_hat, t)    

    return acc_val, iou_val, dice_val


def training_unet(model, wandb, images, masks, optimizer, epochs, feat = 32, num_blocks = 5,
    verbose = True, device = None, validation = True, images_val = None, masks_val = None, path = None, scheduler = None):
    
    # load data
    dataset = Dataset_UNet(images, masks)
    dl = DataLoader(dataset, shuffle = True, batch_size = 64)

    loss = []
    acc = []
    
    ref_iou = 0

    for epoch in tqdm(range(epochs)):
        loss_epoch, acc_epoch, iou_epoch, dice_epoch = training_step_unet(model = model, dl = dl, optimizer = optimizer, device = device)
        loss.append(loss_epoch)
        acc.append(acc_epoch)

        if validation == True:
            acc_val, iou_val, dice_val = validation_unet(model, images_val, masks_val, device = device)

        if verbose == True:
            print("\n####### Epoch ", str(epoch)," #######")
            print("\n## Training ##")
            print("Accuracy: ", np.round(acc_epoch, 5))
            print("IoU: ", np.round(iou_epoch.numpy(), 5)) 
            print("Dice: ", np.round(dice_epoch.numpy(),5)) 
            print("Loss: ", np.round(loss_epoch,5))  
            print("\n## Validation ##")
            print("Accuracy: ", np.round(acc_val.numpy(), 5))
            print("IoU Validation: ", np.round(iou_val.numpy(),5))
            print("Dice Validation: ", np.round(dice_val.numpy(),5))
            #print("Loss Validation: ", np.round(loss_val,5)) 
        
        scheduler.step(loss_epoch)

        if wandb is not None:
            wandb.log({"training_loss": loss_epoch,
                        "training_acc": acc_epoch,
                        "training_iou": iou_epoch,
                        #"val_loss": loss_val,
                        "val_acc": acc_val,
                        "val_iou": iou_val})

        if iou_val.numpy() > ref_iou:
            torch.save(model.state_dict(), path)
            ref_iou = iou_val.numpy()
            print("Model Saved.")

    return model

def pretrain_unet(model, wandb, lungs, val_lungs, feat, num_blocks, epochs = 10, retrain = False, verbose = True, device = None, resolution = 128):

    unet = UNet(feat = feat, num_blocks = num_blocks)
    unet.to(device)
    unet.train()
    path = "model_checkpoints/UNet/"+str(resolution)+"_"+str(feat)+"_"+str(num_blocks)+".pt"

    if exists(path) and (retrain == False):
        unet.load_state_dict(torch.load(path, map_location = device))
        try:
            model.enc_local.feat_ext = unet.feat_ext
        except:
            model.feat_ext = unet.feat_ext
         
        print("UNet Loaded.")
        return model

    else:
        # load data
        masks_train = []
        images_train = []
        masks_val = []
        images_val = []
        
        resolution = int(512/resolution)

        for i in tqdm(lungs):
            mask = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()  
            mask = torch.from_numpy(np.where(mask < 100,  0, 1))[:,::resolution,::resolution]
            masks_train.append(mask)

            img = nib.load("data/nifti/image/case_"+str(i)+".nii.gz").get_fdata()
            img = torch.from_numpy(img)[:,::resolution,::resolution]
            images_train.append(img)

        for i in tqdm(val_lungs):
            mask = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()  
            mask = torch.from_numpy(np.where(mask < 100,  0, 1))[:,::resolution,::resolution]
            masks_val.append(mask)

            img = nib.load("data/nifti/image/case_"+str(i)+".nii.gz").get_fdata()
            img = torch.from_numpy(img)[:,::resolution,::resolution]
            images_val.append(img)

        # compute mean and standard deviation of images
        unet.img_mean = nn.parameter.Parameter(torch.mean(torch.cat(images_train)), requires_grad = False)  
        unet.img_std = nn.parameter.Parameter(torch.std(torch.cat(images_train)), requires_grad = False)  

        # standardize images    
        images_train = [(img-unet.img_mean)/unet.img_std for img in images_train] 
        images_val = [(img-unet.img_mean)/unet.img_std for img in images_val] 

        print("Train UNet:")
        optimizer = torch.optim.Adam(unet.parameters(), lr = 0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = 5)

        unet = training_unet(unet, wandb, images_train, masks_train, optimizer, epochs, feat, num_blocks, 
            images_val = images_val, masks_val = masks_val, device = device, path = path, scheduler = scheduler)
        print("UNet Trained.")
        try:
            model.enc_local.feat_ext = unet.feat_ext
        except:
            model.feat_ext = unet.feat_ext
        print("UNet Loaded.")
        return model

def unet_no_decoder(wandb, num_lungs, val_lungs = [], test_lungs = [], feat = 32, num_blocks = 5, epochs = 10, 
    verbose = True, device = None, resolution = 128, augmentations = True, patience = 10, **_):

    unet = UNet(feat = feat, num_blocks = num_blocks)
    unet.to(device)
    unet.train()
    path = "model_checkpoints/final_models/unet.pt"

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
    _, images_train, masks_train = load_data(train_lungs, 
        train = True, 
        point_resolution = resolution, 
        img_resolution = resolution, 
        return_mask = True, 
        augmentations = augmentations,
        unet = True)

    _, images_val, masks_val = load_data(val_lungs, 
        train = True, 
        point_resolution = resolution, 
        img_resolution = resolution, 
        return_mask = True, 
        augmentations = False,
        unet = True)        

    # compute mean and standard deviation of images
    unet.img_mean = nn.parameter.Parameter(torch.mean(torch.cat(images_train)), requires_grad = False)  
    unet.img_std = nn.parameter.Parameter(torch.std(torch.cat(images_train)), requires_grad = False)  

    # standardize images    
    images_train = [(img-unet.img_mean)/unet.img_std for img in images_train] 
    images_val = [(img-unet.img_mean)/unet.img_std for img in images_val] 

    optimizer = torch.optim.Adam(unet.parameters(), lr = 0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = patience)

    unet = training_unet(unet, wandb = wandb, images = images_train, masks = masks_train, optimizer = optimizer, epochs = epochs, feat = feat, num_blocks = num_blocks,
        verbose = verbose, device = device, validation = True, images_val = images_val, masks_val = masks_val, path = path)

    unet.eval()
    for i, img in enumerate(images_val):
        y_hat = unet(img.float().to(device).unsqueeze(1))
        y_hat = y_hat.squeeze().detach().cpu().numpy()
        get_ply(mask=y_hat,  ply_filename = "dump/unet_lung"+str(i), from_mask = True, resolution = 128, device = device)
        wandb.save("visualization/ply_data/dump/unet_lung_"+str(i)+".ply", base_path = "visualization/ply_data")

    torch.save(unet.state_dict(), path)


class Dataset_UNet(Dataset):
    def __init__(self, X, y, transform = None, feat_extraction = False):
        if feat_extraction == True:
            data = torch.stack((torch.unsqueeze(X,1), torch.unsqueeze(y,1)))
        else:
            data = torch.stack((torch.unsqueeze(torch.cat(X),1), torch.unsqueeze(torch.cat(y),1)))
        self.data = torch.moveaxis(data,0,1)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data[idx, 0]
        y = self.data[idx, 1]
        sample = (X,y)
        if self.transform:
            sample = self.transform(sample)
            
        return sample   





if __name__ == "__main__":

    use_cuda = True
    use_cuda = False if not use_cuda else torch.cuda.is_available()
    device = torch.device('cuda:1' if use_cuda else 'cpu')
    torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device)

    params = {
        "num_lungs" : 332,
        "val_lungs" : [0, 1, 178, 179, 180, 329, 330, 331],
        "test_lungs" : [2, 86, 87, 88, 326, 327, 328],
        "feat"  : 32,
        "num_blocks" : 5,
        "epochs"    : 50,
        "resolution" : 128,
        "augmentations" : True,
    }

    # Train model
    wandb.init(project = "unet", config = params, name = "unet_128")
    #unet_no_decoder(device = device, wandb = wandb, **params)

    # Create PLY files
    unet = UNet(**params)
    unet.load_state_dict(torch.load("model_checkpoints/final_models/unet.pt", map_location = device))
    device = "cpu"
    unet.to(device)
    unet.eval()

    val_lungs = [0,1,178,179,180, 329, 330, 331, 2, 86, 87, 88, 326, 327, 328]
    _, images_val, masks_val = load_data(val_lungs, 
        train = True, 
        point_resolution = 128, 
        img_resolution = 128, 
        return_mask = True, 
        augmentations = False,
        unet = True)   

    images_val = [((img.to(device)-unet.img_mean)/unet.img_std).cpu() for img in images_val] 

    iou_list = []
    dice_list = []
    acc_list = []

    for i, img, mask in zip(val_lungs, images_val, masks_val):
        y_hat = unet(img.float().to(device).unsqueeze(1)).squeeze().detach().cpu()
        y_hat_tmp = torch.round(torch.sigmoid(y_hat)).flatten()
        mask = mask.flatten()
        acc_list.append(np.sum(y_hat_tmp.numpy() == mask.numpy())/len(mask.numpy()))
        iou_list.append(iou(y_hat_tmp, mask).numpy())
        dice_list.append(dice_coef(y_hat_tmp, mask).numpy())
        get_ply(mask=y_hat.numpy(),  ply_filename = "dump/unet_lung_"+str(i), from_mask = True, resolution = 128, device = device)
        wandb.save("visualization/ply_data/dump/unet_lung_"+str(i)+".ply", base_path = "visualization/ply_data")

    results = pd.DataFrame({"Lung": val_lungs,
        "Accuracy" : acc_list,
        "Dice" : dice_list,
        "IoU" : iou_list})
    
    results.to_csv("results/UNet_val_metrics.csv")
    wandb.save("results/UNet_val_metrics.csv")