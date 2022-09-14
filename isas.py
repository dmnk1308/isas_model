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
from torch.profiler import profile, record_function, ProfilerActivity
import wandb

use_cuda = True
use_cuda = False if not use_cuda else torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)

#####################################################
# set seed
torch.manual_seed(123)
np.random.seed(123)
model_name = "test_first_layernorm" 

params = {
# set sizes
"batch_size": 1500,
"batch_size_val": 1500,
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
"num_encoding_functions" : 6,
"num_feat" : 64,        # how many features maps are generated in the first conv layer
"num_feat_attention" : 32, # how many features are used in the self attention layer
"num_feat_out" : 64,
"num_feat_out_xy" : 32,   # how many features are returned in attention layer

# data
"num_lungs" : 50,
"val_lungs" : [0, 1, 2, 3, 4],# 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331],
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
"layer_norm" : False,
"verbose" : True,

# path to model files
"model_name" : model_name,
}

#####################################################
wandb.init(project = "global_model", config = params, name = model_name)
wandb.save("/home/dbecker/masterlung/attention_models.py", policy = "now")

# setup model 
model = GlobalXY(**params)
model.train()
model.to(device)

# training
model, acc, iou_, dice, iou_val = train(model = model, wandb = wandb, visualize_epochs = True, device = device, **params)

# visualize
visualize(model, wandb, np.array([3, 4, 5, 0, 1, 178, 179, 180, 329, 330, 331, 2, 86, 87, 88, 326, 327, 328]), max_batch = 2000, device = device, **params)

# visualize mask
#visualize_mask(327, 332, resolution = 512, device=device)