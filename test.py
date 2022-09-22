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
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)

#####################################################
model_name = "full" 

params = {
# set sizes
"batch_size": 1500,
"batch_size_val": 10000,
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

# data
"num_lungs" : 332,
"val_lungs" : [0, 1, 2, 3, 4, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331],
"test_lungs" : [5, 6, 7, 8, 86, 87, 88, 178, 179, 180, 305, 306, 307, 308, 309, 310, 311, 312, 313],

# resolutions
"point_resolution" : 128,
"img_resolution" : 128,
"shape_resolution" : 128,
"proportion" : 0.8,

# model type
"augmentations": True,
"pos_encoding" : True,
"skips" : True,
"siren" : False,
"spatial_feat" : True,     # use xy-coordinate specific features 
"global_feat" : False,       # pool over z dimension
"layer_norm" : True,
"verbose" : True,

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


################################ TEST EVALUATION #################################################################
# define lungs to visualize
lungs = [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
iou_values = []
dice_values = []
acc_values = []

# load lungs to visualize
_, img_list, mask_list = load_data(lungs, 
    train = False,
    point_resolution =  params["point_resolution"],
    return_mask = True,
    batch_size=params["batch_size_val"],
    )

for i, img, mask in tqdm(zip(lungs, img_list, mask_list)):
    img_raw = img
    mask_raw = mask

    # only every second slice
    slice_index = torch.arange(0,int(img_raw.shape[0]))

    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]

    mask = mask.moveaxis(0,-1)
    pred = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = 2500, slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = torch.round(torch.sigmoid(torch.from_numpy(pred)))
    dice =  dice_coef(pred,mask)
    iou_value = iou(pred, mask)
    acc = torch.sum(pred == mask)/len(mask.flatten())
    
    iou_values.append(iou_value.numpy())
    dice_values.append(dice.numpy())
    acc_values.append(acc.numpy())

print("IoU Test: ", np.mean(np.array(iou_values)))
print("Dice Test: ", np.mean(np.array(dice_values)))
print("Acc Test: ", np.mean(np.array(acc_values)))

################################################################################################


# define lungs to visualize
lungs = [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]

# load lungs to visualize
_, img_list, mask_list = load_data(lungs, 
    train = False,
    point_resolution =  params["point_resolution"],
    return_mask = True,
    batch_size=params["batch_size_val"],
    )

for i, img, mask in zip(lungs, img_list, mask_list):
    img_raw = img
    mask_raw = mask

    # only every second slice
    slice_index = torch.arange(0,int(img_raw.shape[0]),10)
    #slice_index = torch.arange(int(img_raw.shape[0]/2)-10,int(img_raw.shape[0]/2)+10)

    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]
    #mask = mask_raw[slice_index]

    mask = mask.moveaxis(0,-1)
    #pred = model_to_voxel(model,device=device, img = img, resolution = shape_resolution, max_batch = 64 ** 3)    
    pred = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = 2500, slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    #get_ply(mask = pred, ply_filename = "dump/reduced_few_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], 
    #        device = device, slicewise = False, slices = [25,26,27,28])
    
    pred = torch.sigmoid(torch.from_numpy(pred))
    # fig, ax = plt.subplots(figsize=(6,4))    
    # sns.histplot(pred[25].flatten(), bins = 100, ax = ax,  stat="density")
    # plt.savefig("test/lung_"+str(i)+"_ISAS_hist.png")

    # #pred = pred.moveaxis(-1,0)
    # for j, slice in enumerate(pred):
    #     slice = torch.round(slice)
    #     plt.imsave("test/lung_"+str(i)+"_"+str(j)+"_ISAS.png", slice, cmap = "gray")    
    
    pred = pred.moveaxis(0,-1)
    dice =  dice_coef(pred,mask)
    iou_value = iou(pred, mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

    ################################################################

    # middle 10 slices
    slice_index = torch.arange(int(img_raw.shape[0]/2)-10,int(img_raw.shape[0]/2)+10)
    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]
    #mask = mask_raw[slice_index]

    #mask = mask.moveaxis(0,-1)
    #pred = model_to_voxel(model,device=device, img = img, resolution = shape_resolution, max_batch = 64 ** 3)    
    pred = model_to_voxel(model,device=device, img = img, resolution = mask.shape[1], z_resolution= mask.shape[-1], max_batch = 1000, slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_middle_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], device = device)
    pred = torch.from_numpy(pred).moveaxis(0,-1)
    dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
    iou_value = iou(torch.round(torch.sigmoid(pred)), mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

    ################################################################
    slice_index = torch.arange(int(img_raw.shape[0])-20,int(img_raw.shape[0]))
    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]
    #mask = mask_raw[slice_index]

    #mask = mask.moveaxis(0,-1)
    #pred = model_to_voxel(model,device=device, img = img, resolution = shape_resolution, max_batch = 64 ** 3)    
    pred = model_to_voxel(model,device=device, img = img, resolution = mask.shape[1], z_resolution= mask.shape[-1], max_batch = 1000, slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/reduced_bottom_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], device = device)
    pred = torch.from_numpy(pred).moveaxis(0,-1)
    dice =  dice_coef(torch.round(torch.sigmoid(pred)),mask)
    iou_value = iou(torch.round(torch.sigmoid(pred)), mask)

    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

#visualize_mask(50, 332, resolution = 512, device=device)