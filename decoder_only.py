from models import *
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
#device = "cpu"

#####################################################
# set seed
torch.manual_seed(123)
np.random.seed(123)
model_name = "decoder_only" 

params = {
# set sizes
"batch_size": 1500,
"batch_size_val": 50000,
"sample_size" : 150000,

# model parameters
"epochs" : 10,
"lr" : 0.001,
"num_nodes" : 512,              # nodes in decoder
"num_nodes_first" : 512,        # nodes in first layer of decoder
"num_layer" : 6,                # layer of decoder (incl. first one)      
"num_blocks" : 5,               # number of convolutional blocks
"dropout" : 0.0,            
"weight_decay"  : 0.0,
"patience" : 20,                # disabled
"num_encoding_functions" : 10,  # number of fourier transforms for each coordinate (E in paper)
"num_feat" : 256,               # number of latent features per pair of lungs

# data
"num_lungs" : 332,
"val_lungs" : [0, 1, 2, 3, 4, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331],
#"val_lungs" : [],
"test_lungs" : [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304],
#"test_lungs" : [],

# resolutions
"point_resolution" : 128,   # resolution of masks for sampling
"img_resolution" : 128,     # resolution of ct images
"shape_resolution" : 128,   # resolution of reconstructions
"proportion" : 0.5,         # portion of samples from surface of lungs

# model type
"augmentations": False,      # True: enable shift and zoom augmentations
"pos_encoding" : True,      # True: use positional encoding in the decoder
"skips" : True,             # True: use skip conenction in the decoder
"spatial_feat" : True,      # True: use xy features
"global_feat" : False,      # True: use global features (pooling over slice features)
"layer_norm" : True,        # True: use layernorm in the encoder
"batch_norm" : False,       # True: use batchnorm in the encoder
"verbose" : True,           # True: give console output
"pe_freq" : "2",            # set frequency bands of fourier transforms ("2pi" - Mildenhal et al.; "2" - see paper)
"no_encoder" : True,        # True: use training procedure without encoder 

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
#model, acc, iou_, dice, iou_val = train(model = model, wandb = wandb, device = device, **params)

########################################## TEST #####################################################
# load model
model.load_state_dict(torch.load("model_checkpoints/final_models/" + params["model_name"] +".pt", map_location = device))
print("Model loaded.")
model.eval()

lungs = params["test_lungs"]
#lungs = [6,284]

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
    # get_ply(mask = pred, ply_filename = "dump/ISAS_"+model_name+"_lung_"+str(i), from_mask = True, resolution =  params["shape_resolution"], 
    #       device = device)
    
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

results.to_csv("results/TEST_NODECODER_metrics.csv")