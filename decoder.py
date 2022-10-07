# load utils
import sys
import os

from models2 import *
from helpers import *
from render import *
from training import *
 
# load modules
import numpy as np
import torch
import wandb

use_cuda = True
use_cuda = False if not use_cuda else torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)
device = "cpu"

#####################################################
# set seed
torch.manual_seed(123)
np.random.seed(123)
model_name = "decoder_point512" 

params = {
# set sizes
"batch_size": 1500,
"batch_size_val": 50000,
"sample_size" : 150000,

# model parameters
"epochs" : 10,
"lr" : 0.001,
"num_nodes" : 128,
"num_nodes_first" : 128,
"num_layer" : 6,    #6
"num_blocks" : 5,
"dropout" : 0.0,
"weight_decay"  : 0.0,
"patience" : 2,
"num_encoding_functions" : 10,
"num_feat" : 256,        # how many features maps are generated in the first conv layer
"num_feat_attention" : 256, # how many features are used in the self attention layer
"num_feat_out" : 10,   # how many features are returned in attention layer
"num_lungs" : 332,
"val_lungs" : [0, 1, 2, 3, 4, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331],
"test_lungs" : [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304],

# resolutions
"point_resolution" : 128,
"img_resolution" : 128,
"shape_resolution" : 128,
"proportion" : 0.5,

# model type
"augmentations": False,
"pos_encoding" : True,
"skips" : True,
"siren" : False,
"spatial_feat" : True,     # use xy-coordinate specific features 
"global_feat" : False,       # pool over z dimension
"layer_norm" : True,
"batch_norm" : False,
"no_encoder" : True,
"verbose" : True,
"pe_freq" : "2",

# path to model files
"model_name" : model_name,
}

#####################################################
wandb.init(project = "decoder_only", config = params, name = model_name)

# setup model 
model = MLP_with_feat(**params)
model.train()
model.to(device)

# training
model, acc, iou_, dice, iou_val = train(model = model, wandb = wandb, device = device, **params)


########################################## TEST #####################################################
# load model
model.load_state_dict(torch.load("model_checkpoints/final_models/" + params["model_name"] +".pt", map_location = device))
print("Model loaded.")
model.eval()

#lungs = [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304]
lungs = [6,284]

# load lungs to visualize
_, img_list, mask_list = load_data(lungs, 
    train = False,
    point_resolution = 128,
    img_resolution = 128,
    return_mask = True,
    augmentations=False
    )

# save metrics
iou_list = []
dice_list = []
acc_list = []

# loop over lungs
for i, img, mask in zip(lungs, img_list, mask_list):
    img_raw = img
    mask_raw = mask

    slice_index = torch.arange(0,int(img_raw.shape[0]),1)
    slice_max = img_raw.shape[0]

    img = img_raw[slice_index]

    mask = mask.moveaxis(0,-1)
        
    pred = model_to_voxel(model,lung = i, device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max, no_encoder=True)   

    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)
    get_ply(mask = pred, ply_filename = "dump/ISAS_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], 
          device = device)
    
    pred = torch.round(torch.sigmoid(torch.from_numpy(pred)))
    
    # #pred = pred.moveaxis(-1,0)
    # for j, slice in enumerate(pred):
    #     slice = torch.round(slice)
    #     plt.imsave("test/lung_"+str(i)+"_"+str(j)+"_ISAS.png", slice, cmap = "gray")    
    
    pred = pred.moveaxis(0,-1)
    dice =  dice_coef(pred,mask)
    iou_value = iou(pred, mask)
    acc = torch.sum(pred.flatten() == mask.flatten())/len(mask.flatten())

    iou_list.append(iou_value.numpy())
    dice_list.append(dice.numpy())
    acc_list.append(acc.numpy())


    print("Dice Coeff.: ", dice.numpy())
    print("IoU: ",iou_value.numpy())

iou_test = np.mean(np.array(iou_list))
dice_test = np.mean(np.array(dice_list))
acc_test = np.mean(np.array(acc_list))


print("IoU Test: ", iou_test)
print("Dice Test ", dice_test)
print("Acc Test: ", acc_test)

results = pd.DataFrame({"Lung": lungs,
    "Accuracy" : acc_list,
    "Dice" : dice_list,
    "IoU" : iou_list})

results.to_csv("results/TEST_NODECODER_metrics.csv")