import torch
import torch.nn as nn
import torch.nn.functional as F


# load utils
from helpers import *


### SIMPLE MLP ###
class MLP_simple(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = False, 
                 siren = False, 
                 num_encoding_functions = 6,
                 random_sampling = False):
        
        super(MLP_simple, self).__init__()
        self.num_layer = num_layer
        self.pos_enc = pos_encoding
        self.num_encoding_functions = num_encoding_functions
        self.random_sampling = random_sampling 
        pos_dim = num_encoding_functions*2*3
        
        # empty lists for layers
        self.layer = torch.nn.ModuleList()
        
        if pos_encoding == True:
        # define first layer 
            self.layer.append(torch.nn.Linear(3+pos_dim,num_nodes_first))
        else:
            self.layer.append(torch.nn.Linear(3,num_nodes_first))
            
        # define num_layer additional layer      
        for i in range(num_layer-1):
            if i == 0:
                self.layer.append(torch.nn.Linear(num_nodes_first, num_nodes))
            else:
                self.layer.append(torch.nn.Linear(num_nodes, num_nodes))
        
        # define last layer with one output node
        self.layer.append(torch.nn.Linear(num_nodes, 1))

        if siren == True:
            # initialize layer 
            for i in self.layer:
                siren_uniform_(i.weight, mode='fan_in', c=6)
            self.activation0 = Sine(w0 = 30)
            self.activation = Sine(w0=1)
                
        else:
            self.activation0 = torch.nn.ReLU()
            self.activation = torch.nn.ReLU()

        
    def forward(self, x):
        
        if self.pos_enc == True:
            x = positional_encoding(x, num_encoding_functions = self.num_encoding_functions)#, random_sampling = self.random_sampling)
        x = self.layer[0](x)
        x = self.activation0(x)

        for i in self.layer[1:-1]:
            x = i(x)
            x = self.activation(x)
        
        x = self.layer[-1](x)
        
        return x
    
def model_wrapper(dl_train, dl_test, mask = None, pos_encoding = True, siren = True, num_encoding_functions = 6, 
                  num_nodes = 512, num_layer = 4, num_nodes_first = 256,
                  lr = 0.01, epochs = 100, patience = 100, verbose = True, eval_slice = 27, compare = True, random_sampling = False, device = None):

    # initialize model
    model = MLP_simple(num_nodes = num_nodes, num_layer = num_layer, num_nodes_first = num_nodes_first, pos_encoding = pos_encoding, 
                       siren = siren, num_encoding_functions=num_encoding_functions, random_sampling = random_sampling)
    model.to(device)
    model.train()

    # train with binary cross entropy and ADAM optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose = verbose, patience = patience)

    # training
    model, loss, acc = training(model = model, dl = dl_train, optimizer = optimizer, criterion = criterion, scheduler = scheduler, epochs = epochs, verbose = verbose, device = device)

    # plot training
    plot_training_curves(loss, acc)

    # test
    coord_test, y_hat = test(model = model, slice = eval_slice, device = device) 

    # plot test image
    plot_test(model = model, coord = coord_test, y_hat = y_hat, mask = mask, compare = compare, slice = eval_slice)
    
    # get overall accuracy
    # get_acc(model = model, dataloader = dl_test)
    
    return model



########### MLP FEAT 
# class MLP_with_feat(torch.nn.Module):
#     def __init__(self, num_layer = 5, 
#                  num_nodes = 512, 
#                  num_nodes_first = 128, 
#                  pos_encoding = False, 
#                  num_feat = 50,
#                  num_lungs = 10,
#                  siren = False, 
#                  num_encoding_functions = 6,
#                  random_sampling = False):
        
#         super(MLP_with_feat, self).__init__()
#         self.num_layer = num_layer
#         self.pos_enc = pos_encoding
#         self.num_encoding_functions = num_encoding_functions
#         self.random_sampling = random_sampling 
#         pos_dim = num_encoding_functions*2*3
        
#         # initialize parameters for lung fingerprint
#         self.lungfinger = torch.nn.ParameterList()
#         for i in range(num_lungs):
#             self.lungfinger.append(torch.nn.parameter.Parameter(torch.randn(num_feat)))

#         # empty lists for layers
#         self.layer = torch.nn.ModuleList()
        
#         if pos_encoding == True:
#         # define first layer 
#             self.layer.append(torch.nn.Linear(pos_dim+num_feat,num_nodes_first))
#         else:
#             self.layer.append(torch.nn.Linear(3,num_nodes_first))
            
#         # define num_layer additional layer      
#         for i in range(num_layer-1):
#             if i == 0:
#                 self.layer.append(torch.nn.Linear(num_nodes_first+num_feat, num_nodes))
#             else:
#                 self.layer.append(torch.nn.Linear(num_nodes, num_nodes))
        
#         # define last layer with one output node
#         self.layer.append(torch.nn.Linear(num_nodes, 1))

#         if siren == True:
#             # initialize layer 
#             for i in self.layer:
#                 siren_uniform_(i.weight, mode='fan_in', c=6)
#             self.activation0 = Sine(w0 = 30)
#             self.activation = Sine(w0=1)
                
#         else:
#             self.activation0 = torch.nn.ReLU()
#             self.activation = torch.nn.ReLU()

        
#     def forward(self, data):

#         x,z = data
#         if self.pos_enc == True:
#             x = positional_encoding(x, num_encoding_functions = self.num_encoding_functions, include_input=False)#, random_sampling = self.random_sampling)
        
#         feat_list = []#torch.nn.ParameterList()
#        # if z.shape[1] == 1:
#         for i in z:
#             feat_list.append(self.lungfinger[i.int()])
#         z = torch.stack(feat_list)
#         x = torch.cat((x,z), dim = 1)
#         x = self.layer[0](x)
#         x = self.activation0(x)
#         # else:
#         #     for i in z:
#         #         lung1 = self.lungfinger[i[0].int()]
#         #         lung2 = self.lungfinger[i[1].int()]
#         #         feat_list.append((lung1+lung2)/2)
#             # z = torch.stack(feat_list)
#             # x = torch.cat((x,z), dim = 1)
#             # x = self.layer[0](x)
#             # x = self.activation0(x)

#         for i in self.layer[1:-1]:
#             x = torch.cat((x,z), dim = 1)
#             x = i(x)
#             x = self.activation(x)
        
#         x = self.layer[-1](x)
        
#         return x

# define 2d UNet
class Encoder_block(torch.nn.Module):
    def __init__(self, in_feat = 1, out_feat = 32, last_relu = True):
        super(Encoder_block, self).__init__()
        self.last_relu = last_relu
        self.conv1 = nn.Conv2d(in_feat,out_feat,3,padding="same", padding_mode = "reflect")
        self.batch1 = nn.BatchNorm2d(out_feat)
        self.conv2 = nn.Conv2d(out_feat,out_feat ,3,padding="same", padding_mode = "reflect")
        self.batch2 = nn.BatchNorm2d(out_feat)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        if self.last_relu == True:
            x = F.relu(x)  

        return x 

class Encoder_unet(torch.nn.Module):
    def __init__(self, feat = 32, in_channels = 1, num_enc_blocks = 5, last_relu = True):
        super(Encoder_unet, self).__init__()
        self.feat = feat
        self.in_channels = in_channels

        self.enc_blocks = torch.nn.ModuleList()
        for i in range(num_enc_blocks):
            if i == num_enc_blocks-1:
                self.enc_blocks.append(Encoder_block(in_feat = self.in_channels, out_feat = self.feat, last_relu = last_relu))
            else:
                self.enc_blocks.append(Encoder_block(in_feat = self.in_channels, out_feat = self.feat))
                self.in_channels = self.feat
                self.feat = self.in_channels * 2
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        res = []
        for block in self.enc_blocks:
            x = block(x)
            res.append(x)
            x = self.max_pool(x)
        return res

class Decoder_block(torch.nn.Module):
    def __init__(self, feat = 1024, last_relu = True):
        self.last_relu = last_relu
        super(Decoder_block, self).__init__() 
        self.upconv = nn.ConvTranspose2d(feat, feat, kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv2d(feat + int(feat/2), int(feat/2), 3,padding="same", padding_mode = "reflect")
        feat = int(feat/2)
        self.batch1 = nn.BatchNorm2d(feat)
        self.conv2 = nn.Conv2d(feat, feat, 3,padding="same", padding_mode = "reflect")
        self.batch2 = nn.BatchNorm2d(feat)

    def forward(self,x):
        x_curr, x_enc = x
        x_curr = self.upconv(x_curr)
        x = torch.cat((x_enc, x_curr), dim = 1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        if self.last_relu == True:
            x = F.relu(x)
        return x

class Decoder_unet(torch.nn.Module):
    def __init__(self, feat = 1024, num_dec_blocks = 4, last_relu = True):        
        super(Decoder_unet, self).__init__() 
        self.last_relu = last_relu
        self.feat = feat
        self.dec_blocks = torch.nn.ModuleList()

        for i in range(num_dec_blocks):
            if i == num_dec_blocks-1:
                if last_relu == False:
                    self.dec_blocks.append(Decoder_block(feat = self.feat, last_relu = last_relu))
                else:
                    self.dec_blocks.append(Decoder_block(feat = self.feat))
            else:
                self.dec_blocks.append(Decoder_block(feat = self.feat))
            self.feat = int(self.feat/2)


    def forward(self, x):
            for i, block in enumerate(self.dec_blocks):
                x_tmp = block([x[-(i+1)], x[-(i+2)]])
                x[-(i+2)] = x_tmp
            return x[0]

class UNet(torch.nn.Module):
    def __init__(self, feat = 32, num_blocks = 5, **_):
        super(UNet, self).__init__() 
        self.img_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad = False) 
        
        # self.img_mean = nn.parameter.Parameter(torch.Tensor([0.]), requires_grad = False)
        # self.img_std = nn.parameter.Parameter(torch.Tensor([1.]), requires_grad = False) 
        self.feat_ext = FeatExt(feat = feat, num_blocks = num_blocks)
        self.conv_last = nn.Conv2d(feat, 1, 1, padding="same", padding_mode = "reflect")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.feat_ext(x)
        x = F.relu(x)
        x = self.conv_last(x)
        #x = self.sigmoid(x)
        return x

class FeatExt(torch.nn.Module):
    def __init__(self, feat = 32, num_blocks = 5):
        super(FeatExt, self).__init__() 
        self.enc = Encoder_unet(feat = feat, num_enc_blocks = num_blocks)
        self.dec = Decoder_unet(feat = feat*(2**(num_blocks-1)), last_relu=False)
       
    def forward(self,x):
        x = self.enc(x)
        x = self.dec(x)

        return x

class Model_full(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = False, 
                 num_feat = 64,
                 siren = False, 
                 num_encoding_functions = 6,
                 random_sampling = False):
        
        super(Model_full, self).__init__()
        self.num_layer = num_layer
        self.pos_enc = pos_encoding
        self.num_encoding_functions = num_encoding_functions
        self.random_sampling = random_sampling 
        pos_dim = num_encoding_functions*2*3

        # empty lists for layers
        self.layer = torch.nn.ModuleList()
        self.batchnorm = torch.nn.ModuleList()
        
        if pos_encoding == True:
        # define first layer 
            self.layer.append(torch.nn.Linear(3+pos_dim+num_feat,num_nodes_first))
            self.batchnorm.append(torch.nn.BatchNorm1d(num_nodes_first))
        else:
            self.layer.append(torch.nn.Linear(3,num_nodes_first))
            self.batchnorm.append(torch.nn.BatchNorm1d(num_nodes_first))
            
        # define num_layer additional layer      
        for i in range(num_layer-1):
            if i == 0:
                self.layer.append(torch.nn.Linear(num_nodes_first, num_nodes))
                self.batchnorm.append(torch.nn.BatchNorm1d(num_nodes))

            else:
                self.layer.append(torch.nn.Linear(num_nodes, num_nodes))
                self.batchnorm.append(torch.nn.BatchNorm1d(num_nodes))    
        
        # define conv layer
        self.feat_lin = torch.nn.Linear(num_feat,num_feat)

        # define last layer with one output node
        self.layer.append(torch.nn.Linear(num_nodes, 1))

        if siren == True:
            # initialize layer 
            for i in self.layer:
                siren_uniform_(i.weight, mode='fan_in', c=6)
            self.activation0 = Sine(w0 = 30)
            self.activation = Sine(w0=1)
                
        else:
            self.activation0 = torch.nn.ReLU()
            self.activation = torch.nn.ReLU()

        
    def forward(self, data):

        x,z = data
        if self.pos_enc == True:
            x = positional_encoding(x, num_encoding_functions = self.num_encoding_functions)#, random_sampling = self.random_sampling)
        
        z = self.feat_lin(z)
        z = self.activation(z)
        x = torch.cat((x,z), dim = 1)
        x = self.layer[0](x)
        #x = self.batchnorm[0](x)
        x = self.activation0(x)

        for i in zip(self.layer[1:-1], self.batchnorm[1:-1]):
            x = i[0](x)
        #    x = i[1](x)
            x = self.activation(x)
        
        x = self.layer[-1](x)
        
        return x