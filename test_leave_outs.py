from models2 import *
from helpers import *
from render import *
from training import *
 
# load modules
import numpy as np
import torch
import wandb
import seaborn as sns


use_cuda = True
use_cuda = False if not use_cuda else torch.cuda.is_available()
device = torch.device('cuda:1' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)
device = "cpu"
#####################################################
model_name = "full_100k_unbalanced" 

params = {
# set sizes
"batch_size": 1500,
"batch_size_val": 3000,
"sample_size" : 150000,

# model parameters
"epochs" : 5,
"lr" : 0.00005,
"num_nodes" : 512,
"num_nodes_first" : 512,
"num_layer" : 6,    #6
"num_blocks" : 5,
"dropout" : 0.0,
"weight_decay"  : 0.0,
"patience" : 2,
"num_encoding_functions" : 10,
"num_feat" : 64,        # how many features maps are generated in the first conv layer
"num_feat_attention" : 32, # how many features are used in the self attention layer
"num_feat_out" : 256,
"num_feat_out_xy" : 32,   # how many features are returned in attention layer

# resolutions
"point_resolution" : 128,
"img_resolution" : 128,
"shape_resolution" : 128,
"proportion" : 0.8,
"get_weights" : False,

# model type
"pos_encoding" : True,
"skips" : True,
"siren" : False,
"spatial_feat" : True,     # use xy-coordinate specific features 
"global_feat" : False,       # pool over z dimension
"layer_norm" : True,
"verbose" : True,
"pe_freq" : "2",

# path to model files
"model_name" : model_name,
}
#####################################################

# setup model 
model = ISAS(**params)
model.to(device)

# load
model.load_state_dict(torch.load("model_checkpoints/final_models/" + params["model_name"] +".pt", map_location = device))
print("Model loaded.")

# visualize
model.eval()

# define lungs to visualize
lungs = [6,284]

# load lungs to visualize
_, img_list, mask_list = load_data(lungs, 
    train = False,
    point_resolution =  params["point_resolution"],
    return_mask = True,
    batch_size=params["batch_size_val"],
    )

# save metrics
iou_list = []
dice_list = []
acc_list = []

# loop over lungs
for i, img, mask in zip(lungs, img_list, mask_list):
    img_raw = img
    mask_raw = mask

############################## 10th #############################
    slice_index = torch.arange(0,int(img_raw.shape[0]),10)
    slice_max = img_raw.shape[0]
    img = img_raw[slice_index]

    mask = mask.moveaxis(0,-1)
    if params["get_weights"] == True:
        pred, att = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max, get_weights = True)   
        np.save("att_lung_"+str(i)+".npy", att)

    else:
        pred = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_few_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], 
            device = device)
    
    pred = torch.round(torch.sigmoid(torch.from_numpy(pred)))
    pred = pred.moveaxis(0,-1)

    dice =  dice_coef(pred,mask)
    iou_value = iou(pred, mask)
    acc = torch.sum(pred.flatten() == mask.flatten())/len(mask.flatten())

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

################################################################

############################## middle 20 #############################
    slice_index = torch.arange(int(img_raw.shape[0]/2)-10,int(img_raw.shape[0]/2)+10)
    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]

    if params["get_weights"] == True:
        pred, att = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max, get_weights = True)   
        np.save("att_middle_lung_"+str(i)+".npy", att)

    else:
        pred = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max)   
    
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_middle_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], device = device)
    pred = torch.from_numpy(pred).moveaxis(0,-1)
    dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
    iou_value = iou(torch.round(torch.sigmoid(pred)), mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

################################################################

############################## bottom 20 #############################
    slice_index = torch.arange(int(img_raw.shape[0])-20,int(img_raw.shape[0]))
    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]
    
    if params["get_weights"] == True:
        pred, att = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max, get_weights = True)   
        np.save("att_bottom_lung_"+str(i)+".npy", att)

    else:
        pred = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max)   
    
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_bottom_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], device = device)
    pred = torch.from_numpy(pred).moveaxis(0,-1)
    dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
    iou_value = iou(torch.round(torch.sigmoid(pred)), mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

############################## top 20 #############################
    slice_index = torch.arange(20)
    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]
    
    if params["get_weights"] == True:
        pred, att = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max, get_weights = True)   
        np.save("att_bottom_lung_"+str(i)+".npy", att)

    else:
        pred = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max)   
    
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_top_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], device = device)
    pred = torch.from_numpy(pred).moveaxis(0,-1)
    dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
    iou_value = iou(torch.round(torch.sigmoid(pred)), mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

