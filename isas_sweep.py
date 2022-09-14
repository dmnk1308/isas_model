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

start_sweep = False

if start_sweep == True:
    model_name = "test_sample_batch_size" 

    sweep_config = {
        "name": "test_sample_batch_size",
        "method": "grid", #"random",
        "metric": {
            "name" : "training_loss",
            "goal" : "minimize"
            }
    }

    params = {

    # set sizes
    "batch_size": {
        "values": [1500, 1000]
        },
    "batch_size_val": {
        "value": 1500,
        },
    "sample_size" : {
        "values": [50000, 100000, 150000],
        },

    # model parameters
    "epochs" : {
        "value": 5
        },
    "lr" : {
        # "distribution" : "uniform",
        # "min": 0.00001,
        # "max": 0.001
        "value": 0.00005
        },
    "num_nodes" : {
        "value": 512
        },
    "num_nodes_first" : {
        "value": 512
        },
    "num_layer" : {
        "value": 6
        },    
    "num_blocks" : {
        "value": 5
        },
    "dropout" : {
        "value": 0.0
        },
    "weight_decay"  : {
        "value": 0.0
        },
    "patience" : {
        "value": 2
        },
    "num_encoding_functions" : {
        "value": 6
        },
    "num_feat" : {
        "value": 64
        },        # how many features maps are generated in the first conv layer
    "num_feat_attention" : {
        "value": 32,
        }, # how many features are used in the self attention layer
    "num_feat_out" : {
        "value": 64
        },   # how many features are returned in attention layer
    "num_feat_out_xy" : {
        "value": 32
        },  
    "num_lungs" : {
        "value": 50
        },
    "val_lungs" : {
        "value":[0, 1, 2, 3, 4]#, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331]
        },
    "test_lungs" : {
        "value": [5, 6, 7, 8, 86, 87, 88, 178, 179, 180, 305, 306, 307, 308, 309, 310, 311, 312, 313]
        },
    "latent_dim" : {
        "value": 32
        }, 

    # resolutions
    "point_resolution" : {
        "value": 128
        },
    "img_resolution" : {
        "value": 128
        },
    "shape_resolution" : {
        "value": 128
        },
    "proportion" : {
        "value": 0.8
        },

    # model type
    "augmentations": {
        "value": True
        },
    "pos_encoding" : {
        "value": True
        },
    "skips" : {
        "value": True
        },
    "siren" : {
        "value": False
        },
    "spatial_feat" : {
        "value": True
        },
    "global_feat" : {
        "value": False
        },
    "layer_norm" : {
        "value": False
        },
    "verbose" : {
        "value": True
        },

    # path to model files
    "model_name" : {
        "value": model_name
        },
    "batch_norm": {
        "value" : True
        }
    }

    sweep_config["parameters"] = params
    sweep_id = wandb.sweep(sweep_config, project = "global_model")

else:
    sweep_id = "1mh0aezf"

#####################################################
def train_model(config = None):
    with wandb.init(config = config) as run:
        # set seed
        torch.manual_seed(123)
        np.random.seed(123)

        use_cuda = True
        use_cuda = False if not use_cuda else torch.cuda.is_available()
        device = torch.device('cuda:0' if use_cuda else 'cpu')
        torch.cuda.get_device_name(device) if use_cuda else 'cpu'
        print('Using device', device)

        config = wandb.config

        # setup model 
        model = GlobalXY(**config)
        model.train()
        model.to(device)

        # training
        model, acc, iou_, dice, iou_val = train(model = model, wandb = wandb, device = device, **config)
        #wandb.save("attention_models.py")

        # visualize
        visualize(model, wandb, np.array([0,1,2, 178, 179, 180, 329,330,331]), device = device, **config)

#####################################################

wandb.agent("dmnk/global_model/"+sweep_id, function = train_model, count=5)