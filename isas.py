from models import *
from helpers import *
from render import *
from training import *
 
# load modules
import random
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
random.seed(123)

# name model run
model_name = "50k_border_unbalanced_test2"

# set model parameters
params = {
# set sizes
"batch_size": 500,
"batch_size_val": 1400,     
"sample_size" : 50000,
"random" : False,               # True: balanced sampling
"unbalanced" : True,            # True: unbalanced sampling
"border" : True,                # True: focus on surface points for sampling (portion below)

# model parameters
"epochs" : 5,           
"lr" : 0.00005,
"num_nodes" : 512,              # nodes in decoder
"num_nodes_first" : 512,        # nodes in first layer of decoder
"num_layer" : 6,                # layer of decoder (incl. first one)      
"num_blocks" : 5,               # number of convolutional blocks
"dropout" : 0.0,            
"weight_decay"  : 0.0,
"patience" : 20,                # disabled
"num_encoding_functions" : 10,  # number of fourier transforms for each coordinate (E in paper)
"num_feat" : 64,                # output channel of first convolutional block (doubles in the subsequent blocks)
"num_feat_attention" : 32,      # channel dimension for self attention layer
"num_feat_out" : 256,           # final channel dimension for slice features
"num_feat_out_xy" : 32,         # final channel dimension for xy features

# data
"num_lungs" : 332,
"val_lungs" : [0, 1, 2, 3, 4, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331],
"test_lungs" : [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304],

# resolutions
"point_resolution" : 128,   # resolution of masks for sampling
"img_resolution" : 128,     # resolution of ct images
"shape_resolution" : 128,   # resolution of reconstructions
"proportion" : 0.5,         # portion of samples from surface of lungs

# model type
"augmentations": True,      # True: enable shift and zoom augmentations
"pos_encoding" : True,      # True: use positional encoding in the decoder
"skips" : True,             # True: use skip conenction in the decoder
"spatial_feat" : True,      # True: use xy features
"global_feat" : False,      # True: use global features (pooling over slice features)
"layer_norm" : True,        # True: use layernorm in the encoder
"batch_norm" : False,       # True: use batchnorm in the encoder
"verbose" : True,           # True: give console output
"pe_freq" : "2",            # set frequency bands of fourier transforms ("2pi" - Mildenhal et al.; "2" - see paper)

# path to model files
"model_name" : model_name,
}

#####################################################
wandb.init(project = "global_model", config = params, name = model_name)
wandb.save("/home/dbecker/masterlung/attention_models.py", policy = "now")

# setup model 
model = ISAS(**params)
model.train()
model.to(device)

# training
model, acc, iou_, dice, iou_val = train(model = model, wandb = wandb, visualize_epochs = False, device = device, **params)