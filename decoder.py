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
"num_layer" : 4,    #6
"num_blocks" : 5,
"dropout" : 0.0,
"weight_decay"  : 0.0,
"patience" : 2,
"num_encoding_functions" : 10,
"num_feat" : 128,        # how many features maps are generated in the first conv layer
"num_feat_attention" : 256, # how many features are used in the self attention layer
"num_feat_out" : 10,   # how many features are returned in attention layer
"num_lungs" : 332,
"val_lungs" : [0,1],#, 178, 179, 180, 329, 330, 331],
"test_lungs" : [],

# resolutions
"point_resolution" : 128,
"img_resolution" : 128,
"shape_resolution" : 128,

# model type
"augmentations": False,
"pos_encoding" : True,
"skips" : True,
"siren" : False,
"spatial_feat" : False,
"vae" : False,
"masked_attention" : False,
"keep_spatial" : False,
"verbose" : True,
"use_weights" : False,
"no_encoder" : True,
"visualize_epochs" : False,

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
wandb.save("/home/dbecker/masterlung/attention_models.py")

# visualize
visualize(model, wandb, np.array([0,1,2,3,4,5,6,7,178,179,180]), normalize=False, device = device, **params)

# visualize mask
#visualize_mask(327, 332, resolution = 512, device=device)