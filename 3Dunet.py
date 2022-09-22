from random import shuffle
import torch 
import numpy
from os.path import exists
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from helpers import *
from models2 import *
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
        #image = image.moveaxis(1,2)
        y_hat = model(image.float().to(device).unsqueeze(0).unsqueeze(0))
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
    dataset = Dataset_UNet3D(images, masks)
    dl = DataLoader(dataset, shuffle = True, batch_size = 6)

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

def unet_no_decoder(wandb, num_lungs, val_lungs = [], test_lungs = [], feat = 32, num_blocks = 5, epochs = 10, 
    verbose = True, device = None, resolution = 128, augmentations = True, patience = 10, **_):

    unet = UNet_3D(feat = feat, num_blocks = num_blocks)
    unet.to(device)
    unet.train()
    path = "model_checkpoints/final_models/unet3d.pt"

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
        unet = True,
        unet3D= True)

    _, images_val, masks_val = load_data(val_lungs, 
        train = True, 
        point_resolution = resolution, 
        img_resolution = resolution, 
        return_mask = True, 
        augmentations = False,
        unet = True, 
        unet3D = True)        

    # compute mean and standard deviation of images
    unet.img_mean = nn.parameter.Parameter(torch.mean(torch.cat(images_train)), requires_grad = False)  
    unet.img_std = nn.parameter.Parameter(torch.std(torch.cat(images_train)), requires_grad = False)  

    # standardize images    
    images_train = [(img-unet.img_mean)/unet.img_std for img in images_train] 
    images_val = [(img-unet.img_mean)/unet.img_std for img in images_val] 

    optimizer = torch.optim.Adam(unet.parameters(), lr = 0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = True, patience = patience)

    unet = training_unet(unet, wandb = wandb, images = images_train, masks = masks_train, optimizer = optimizer, epochs = epochs, feat = feat, num_blocks = num_blocks,
        verbose = verbose, device = device, validation = True, images_val = images_val, masks_val = masks_val, path = path, scheduler = scheduler)

    unet.eval()
    for i, img in enumerate(images_val):
        y_hat = unet(img.float().to(device).unsqueeze(1))
        y_hat = y_hat.squeeze().detach().cpu().numpy()
        get_ply(mask=y_hat,  ply_filename = "dump/unet_lung"+str(i), from_mask = True, resolution = 128, device = device)
        wandb.save("visualization/ply_data/dump/unet_lung_"+str(i)+".ply", base_path = "visualization/ply_data")

    torch.save(unet.state_dict(), path)


class Dataset_UNet3D(Dataset):
    def __init__(self, X, y, transform = None):
        X = torch.stack(X,0)
        y = torch.stack(y,0)
        data = torch.stack((X, y),0)
        self.data = torch.moveaxis(data,0,1)
        self.transform = transform
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = self.data[idx, 0].unsqueeze(0)
        y = self.data[idx, 1]
        sample = (X,y)
        if self.transform:
            sample = self.transform(sample)
            
        return sample   





if __name__ == "__main__":

    use_cuda = True
    use_cuda = False if not use_cuda else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device)
    #device = "cpu"

    params = {
        "num_lungs" : 103,
        "val_lungs" : [0, 1, 2, 3, 4, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331],
        "test_lungs" : [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        "feat"  : 64,
        "num_blocks" : 5,
        "epochs"    : 50,
        "resolution" : 128,
        "augmentations" : True,
    }

    # Train model
    wandb.init(project = "unet3D", config = params, name = "unet_103")
    unet_no_decoder(device = device, wandb = wandb, **params)
    exit()
    # Create PLY files
    unet = UNet(**params)
    unet.load_state_dict(torch.load("model_checkpoints/final_models/unet.pt", map_location = device))
    device = "cpu"
    unet.to(device)
    unet.eval()

    val_lungs = [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304]
    #val_lungs = [178]
    _, images_val, masks_val = load_data(val_lungs, 
        train = True, 
        point_resolution = 128, 
        img_resolution = 512, 
        return_mask = True, 
        augmentations = False,
        unet = True)   

    for img, i in zip(images_val, val_lungs):
        for j, slice in enumerate(img.numpy()):
            plt.imsave("mask_comp/lung_"+str(i)+"_"+str(j)+"_img.png", slice, cmap = "gray")
    

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
        mask_tmp = mask.flatten()
        acc_list.append(np.sum(y_hat_tmp.numpy() == mask_tmp.numpy())/len(mask_tmp.numpy()))
        iou_list.append(iou(y_hat_tmp, mask_tmp).numpy())
        dice_list.append(dice_coef(y_hat_tmp, mask_tmp).numpy())
        get_ply(mask=y_hat.numpy(),  ply_filename = "dump/unet_lung_"+str(i), from_mask = True, resolution = 128, device = device)
        wandb.save("visualization/ply_data/dump/unet_lung_"+str(i)+".ply", base_path = "visualization/ply_data")

        pred = torch.round(torch.sigmoid(y_hat)).numpy()

        for j, slice in enumerate(pred):
            plt.imsave("mask_comp/lung_"+str(i)+"_"+str(j)+"_UNet.png", slice, cmap = "gray")
        for j, slice in enumerate(mask.numpy()):
            plt.imsave("mask_comp/lung_"+str(i)+"_"+str(j)+"_mask.png", slice, cmap = "gray")

    results = pd.DataFrame({"Lung": val_lungs,
        "Accuracy" : acc_list,
        "Dice" : dice_list,
        "IoU" : iou_list})
    
    results.to_csv("results/UNet_val_metrics.csv")
    wandb.save("results/UNet_val_metrics.csv")