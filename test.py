from models import *
from helpers import *
from mesh_render import *
from attention_helpers import *
from attention_models import *
 
# load modules
import numpy as np
import torch
import wandb


use_cuda = True
use_cuda = False if not use_cuda else torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)

#####################################################
model_name = "random_points_15epochs" 

params = {
# set sizes
"batch_size": 500,
"sample_size" : 50000,

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
"num_encoding_functions" : 6,
"num_feat" : 64,        # how many features maps are generated in the first conv layer
"num_feat_attention" : 32, # how many features are used in the self attention layer
"num_feat_out" : 256,
"num_feat_out_xy" : 32,   # how many features are returned in attention layer
"num_lungs" : 50,
"val_lungs" : [0, 1, 2, 3, 323, 324, 325, 326, 327, 328, 329, 330, 331],
"test_lungs" : [4, 5, 6, 86, 87, 88, 178, 179, 180, 320, 321, 322],
"latent_dim" : 32, 

# resolutions
"point_resolution" : 512,
"img_resolution" : 128,
"shape_resolution" : 128,

# model type
"augmentations": True,
"pos_encoding" : True,
"skips" : True,
"siren" : False,
"spatial_feat" : False,
"xy_feat" : True,
"masked_attention" : True,
"keep_spatial" : False,
"verbose" : True,
"use_weights" : False,
"batch_norm" : True,

# path to model files
"model_name" : model_name,
}

#####################################################

# setup model 
model = GlobalXY(**params)
model.to(device)

# load
model.load_state_dict(torch.load("model_checkpoints/final_models/" + params["model_name"] +".pt", map_location = device))
print("Model loaded.")

# visualize
model.eval()

# define lungs to visualize
lungs = [0,1,2]

# load lungs to visualize
_, img_list, mask_list = load_data(lungs, 
    train = False,
    point_resolution =  params["point_resolution"],
    return_mask = True
    )

for i, img, mask in zip(lungs, img_list, mask_list):
    img_raw = img
    mask_raw = mask

    # only every second slice
    slice_index = torch.arange(0,int(img_raw.shape[0]))
    print(slice_index)

    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]
    #mask = mask_raw[slice_index]

    mask = mask.moveaxis(0,-1)
    #pred = model_to_voxel(model,device=device, img = img, resolution = shape_resolution, max_batch = 64 ** 3)    
    pred = model_to_voxel(model,device=device, img = img, resolution = 128, z_resolution= 128, max_batch = 50000, spatial_feat = params["spatial_feat"], 
        xy_feat =  params["xy_feat"], latent_dim =  params["latent_dim"], keep_spatial =  params["keep_spatial"], slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_4_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], device = device)
    continue
    pred = torch.from_numpy(pred).moveaxis(0,-1)
    dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
    iou_value = iou(torch.round(torch.sigmoid(pred)), mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

    ################################################################

    # middle 10 slices
    slice_index = torch.arange(int(img_raw.shape[0]/2)-2,int(img_raw.shape[0]/2)+2)
    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]
    #mask = mask_raw[slice_index]

    #mask = mask.moveaxis(0,-1)
    #pred = model_to_voxel(model,device=device, img = img, resolution = shape_resolution, max_batch = 64 ** 3)    
    pred = model_to_voxel(model,device=device, img = img, resolution = mask.shape[1], z_resolution= mask.shape[-1], max_batch = 50000, spatial_feat = params["spatial_feat"], 
        xy_feat =  params["xy_feat"], latent_dim =  params["latent_dim"], keep_spatial =  params["keep_spatial"], slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_10_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], device = device)
    pred = torch.from_numpy(pred).moveaxis(0,-1)
    dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
    iou_value = iou(torch.round(torch.sigmoid(pred)), mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

    ################################################################
    slice_index = torch.arange(int(img_raw.shape[0])-10,int(img_raw.shape[0]))
    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]
    #mask = mask_raw[slice_index]

    #mask = mask.moveaxis(0,-1)
    #pred = model_to_voxel(model,device=device, img = img, resolution = shape_resolution, max_batch = 64 ** 3)    
    pred = model_to_voxel(model,device=device, img = img, resolution = mask.shape[1], z_resolution= mask.shape[-1], max_batch = 50000, spatial_feat = params["spatial_feat"], 
        xy_feat =  params["xy_feat"], latent_dim =  params["latent_dim"], keep_spatial =  params["keep_spatial"], slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_20_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], device = device)
    pred = torch.from_numpy(pred).moveaxis(0,-1)
    dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
    iou_value = iou(torch.round(torch.sigmoid(pred)), mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

#visualize_mask(50, 332, resolution = 512, device=device)