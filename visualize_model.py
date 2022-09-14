from models import *
from helpers import *
from render import *
from training import *
from models import *
 
# load modules
import numpy as np
import torch
import wandb

use_cuda = True
use_cuda = False if not use_cuda else torch.cuda.is_available()
device = torch.device('cuda:1' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)

#####################################################
# set seed
torch.manual_seed(123)
np.random.seed(123)
model_name = "test_masked_att_60_331" 

params = {
# set sizes
"batch_size": 1000,
"sample_size" : 100000,

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
"num_feat_out" : 60,   # how many features are returned in attention layer
"num_lungs" : 50,
"val_lungs" : [0, 1, 2, 3, 323, 324, 325, 326, 327, 328, 329, 330, 331],
"test_lungs" : [4, 5, 6, 86, 87, 88, 178, 179, 180, 320, 321, 322],
"latent_dim" : 32, 

# resolutions
"point_resolution" : 128,
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

# load model
model = GlobalXY(**params)
model.load_state_dict(torch.load("model_checkpoints/final_models/" + params["model_name"] +".pt", map_location = device))
model.to(device)
model.eval()
print("Model loaded.")

# visualize
visualize(model, wandb, np.array([0, 1, 2, 3, 4, 5, 178, 179, 180, 329, 330, 331, 2, 86, 87, 88, 326, 327, 328]), max_batch = 10000, device = device, **params)
