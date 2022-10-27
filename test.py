from models import *
from helpers import *
from render import *
from training import *
 
# load modules
import numpy as np
import torch
import wandb
import seaborn as sns
from tqdm import tqdm


use_cuda = True
use_cuda = False if not use_cuda else torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)
#device = "cpu"
#####################################################
model_name = "ISAS_model" 

params = {
# set sizes
"batch_size": 1500,
"batch_size_val":3000,
"sample_size" : 150000,

# model parameters
"num_nodes" : 512,
"num_nodes_first" : 512,
"num_layer" : 6,    #6
"num_blocks" : 5,
"dropout" : 0.0,
"weight_decay"  : 0.0,
"num_encoding_functions" : 10,
"num_feat" : 64,        # how many features maps are generated in the first conv layer
"num_feat_attention" : 32, # how many features are used in the self attention layer
"num_feat_out" : 256,
"num_feat_out_xy" : 32,   # how many features are returned in attention layer

# data
"test_lungs" : [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304],

# resolutions
"point_resolution" : 128,
"img_resolution" : 128,
"shape_resolution" : 128,
"proportion" : 0.5,
"get_weights" : True,

# model type
"augmentations": True,
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

model.load_state_dict(torch.load("model_checkpoints/final_models/" + params["model_name"] +".pt", map_location = device))
print("Model loaded.")

# test
model.eval()
#lungs = params["test_lungs"]
lungs = [284,6]

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
for i, img, mask in tqdm(zip(lungs, img_list, mask_list)):

    img_raw = img
    mask_raw = mask

    slice_index = torch.arange(0,int(img_raw.shape[0]),1)
    slice_max = img_raw.shape[0]

    mask = mask.moveaxis(0,-1)
    if params["get_weights"] == True:
        pred, att = model_to_voxel(model,device=device, img = img,resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max, get_weights = True)   

    else:
        pred = model_to_voxel(model,device=device, img = img, resolution =  mask.shape[1], z_resolution= mask.shape[2], max_batch = params["batch_size_val"], slice_index = slice_index, slice_max = slice_max)   
    pred = pred.cpu().numpy()
    pred = np.moveaxis(pred,-1,0)

    level = 0.0
    
    get_ply(mask = pred, ply_filename = "dump/ISAS_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], level = level, device = device)
    pred = torch.round(torch.sigmoid(torch.from_numpy(pred)))  
    
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

#results.to_csv("results/TEST_ISAS_metrics.csv")