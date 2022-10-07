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
    model_name = "feat_out" 

    sweep_config = {
        "name": "feat_out",
        "method": "grid", #"random",
        "metric": {
            "name" : "training_loss",
            "goal" : "minimize"
            }
    }

    params = {

    # set sizes
    "batch_size": {
        "value": 1500
        },
    "batch_size_val": {
        "value": 1500
        },
    "sample_size" : {
        "value": 150000
        },

    # model parameters
    "epochs" : {
        "value": 4
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
        "value": 10
        },
    "num_feat" : {
        "value": 64
        },        # how many features maps are generated in the first conv layer
    "num_feat_attention" : {
        "value": 32,
        }, # how many features are used in the self attention layer
    "num_feat_out" : {
        "values": [64, 128, 256]
        },   # how many features are returned in attention layer
    "num_feat_out_xy" : {
        "values":[16,32,64]
        },  
    "num_lungs" : {
        "value": 103 # 10 healthy + 50 diseased
        },
    "val_lungs" : {
        "value":[0, 1, 2, 3, 4, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331]
        },
    "test_lungs" : {
        "value": [5, 6, 7, 8, 9, 86, 87, 88, 178, 179, 180, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
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
        "value": True
        },
    "batch_norm" : {
        "value": False
        },
    "verbose" : {
        "value": True
        },
    "pe_freq" : {
        "value":"2pi",
        },
    # path to model files
    "model_name" : {
        "value": model_name
        },
    }

    sweep_config["parameters"] = params
    sweep_id = wandb.sweep(sweep_config, project = "global_model")

else:
    sweep_id = "yqbe7db6"

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
        model = ISAS(**config)
        model.train()
        model.to(device)

        # training
        model, acc, iou_, dice, iou_val = train(model = model, wandb = wandb, device = device, **config)
        #wandb.save("attention_models.py")

        # visualize
        visualize(model, wandb, np.array([0,1,2, 178, 179, 180, 329,330,331]), device = device, **config)

#####################################################

wandb.agent("dmnk/global_model/"+sweep_id, function = train_model, count=5)