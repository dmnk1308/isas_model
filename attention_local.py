# load utils
import sys
import os

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
device = torch.device('cuda:1' if use_cuda else 'cpu')
torch.cuda.get_device_name(device) if use_cuda else 'cpu'
print('Using device', device)

#####################################################
# set seed
torch.manual_seed(123)
np.random.seed(123)
model_name = "local_model_final" 

params = {
# set sizes
"batch_size": 5000,
"sample_size" : 100000,

# model parameters
"epochs" : 10,
"lr" : 0.00005,
"num_nodes" : 512,
"num_nodes_first" : 512,
"num_layer" : 3,    #6
"num_blocks" : 5,
"dropout" : 0.0,
"weight_decay"  : 0.0,
"patience" : 2,
"num_encoding_functions" : 6,
"num_feat" : 32,        # how many features maps are generated in the first conv layer
"num_feat_attention" : 256, # how many features are used in the self attention layer
"num_feat_out" : 32,   # how many features are returned in attention layer
"num_lungs" : 332,
"val_lungs" : [0, 1, 178, 179, 180, 329, 330, 331],
"test_lungs" : [2, 86, 87, 88, 326, 327, 328],
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
"vae" : False,
"masked_attention" : False,
"keep_spatial" : False,
"verbose" : True,
"use_weights" : False,
"batch_norm" : False,

# path to model files
"model_name" : model_name,
}

#####################################################
wandb.init(project = "global_model", config = params, name = model_name)

# setup model 
model = GlobalNet(**params)
model.train()
model.to(device)

# training
model, acc, iou_, dice, iou_val = train(model = model, wandb = wandb, visualize_epochs = True, device = device, **params)
wandb.save("/home/dbecker/masterlung/attention_models.py")

# visualize
visualize(model, wandb, np.array([3, 4, 5, 0, 1, 178, 179, 180, 329, 330, 331, 2, 86, 87, 88, 326, 327, 328]), max_batch = 5000, device = device, **params)

# visualize mask
#visualize_mask(327, 332, resolution = 512, device=device)