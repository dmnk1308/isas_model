# load utils
from helpers import *
from render import *
import torch.nn.functional as F 

#from end2end_helpers import *
#from Resnet import *


class Self_attention_block(torch.nn.Module):
    def __init__(self,
                num_feat_in = 256, 
                num_feat = 32,
                pos_enc = False,
                dropout = 0.1,
                pe_freq = "2pi"):
        super(Self_attention_block, self).__init__()
        self.num_feat = num_feat
        self.pos_enc = pos_enc
        self.pe_freq = pe_freq

        # feature reduction
        self.feat_reduction = torch.nn.Linear(num_feat_in, num_feat)

        # define self attention layer
        self.attention = torch.nn.MultiheadAttention(num_feat, 8, dropout = dropout)

        # define linear layers for query, key and values of self attention
        self.query_lin = torch.nn.Linear(num_feat,num_feat)
        self.key_lin = torch.nn.Linear(num_feat,num_feat)
        self.value_lin = torch.nn.Linear(num_feat,num_feat)

        self.layer_norm1 = torch.nn.LayerNorm(num_feat)
        self.layer_norm2 = torch.nn.LayerNorm(num_feat)

        self.feat_lin1 = torch.nn.Linear(num_feat,512)
        self.feat_lin2 = torch.nn.Linear(512,num_feat)

        # feature extension
        self.feat_extentension = torch.nn.Linear(num_feat, num_feat_in)
        self.layer_norm3 = torch.nn.LayerNorm(num_feat_in)

        # ReLU activation
        self.activation = torch.nn.ReLU()

    def forward(self, z_enc, slice_index = None, slice_max = None):

        if slice_max is None:
            slice_max = z_enc.shape[0]
        if slice_index is None:
            slice_index = torch.arange(z_enc.shape[0])

        z_enc_tmp = self.feat_reduction(z_enc)  

        if self.pos_enc == True:
            # positional encoding
            slice_pe = slice_index.unsqueeze(1)
            slice_pe = 2*(slice_pe/(slice_max)) - 1
            num_enc = z_enc_tmp.shape[1]

            slice_pe = positional_encoding(slice_pe.to(z_enc.device), int(num_enc/2), include_input = False, pe_freq = self.pe_freq)
            slice_pe = slice_pe.reshape(z_enc_tmp.shape[0], -1)
            
            z_enc_tmp = z_enc_tmp + slice_pe

        query = self.query_lin(z_enc_tmp)
        key = self.key_lin(z_enc_tmp)
        value = self.value_lin(z_enc_tmp)

        z_att = self.attention(query, key, value, need_weights = False)[0]
        z_att = self.layer_norm1(z_att+z_enc_tmp)
        z_att = z_att+z_enc_tmp

        z = self.feat_lin1(z_att)
        z = self.activation(z)
        z = self.feat_lin2(z)
        z = self.layer_norm2(z + z_att)
            
        z = self.feat_extentension(z)
        z = self.layer_norm3(z + z_enc)
        z = self.activation(z)

        return z

class Attention_block(torch.nn.Module):
    def __init__(self,
                num_feat = 256, 
                pos_encoding = True,
                dropout = 0.1,
                pe_freq = "2pi"):

        super(Attention_block, self).__init__()

        self.pos_encoding = pos_encoding
        self.pe_freq = pe_freq

        # define self attention layer
        self.attention = torch.nn.MultiheadAttention(num_feat, 8, dropout = 0.0, batch_first = True)

        # define linear layers for query, key and values of self attention
        self.query_lin = torch.nn.Linear(num_feat,num_feat)
        self.key_lin = torch.nn.Linear(num_feat,num_feat)
        self.value_lin = torch.nn.Linear(num_feat,num_feat)

        self.layer_norm1 = torch.nn.LayerNorm(num_feat)
        self.layer_norm2 = torch.nn.LayerNorm(num_feat)

        self.feat_lin1 = torch.nn.Linear(num_feat,512)
        self.feat_lin2 = torch.nn.Linear(512,num_feat)

        self.activation = torch.nn.ReLU()


    def forward(self, data, slice_index = None, slice_max = None):
        z_in, x_in = data

        if slice_max is None:
            slice_max = z_in.shape[0]
        if slice_index is None:
            slice_index = torch.arange(z_in.shape[0])

        if self.pos_encoding == True:
            # positional encoding
            slice_pe = slice_index.unsqueeze(1)
            slice_pe = 2*(slice_pe/(slice_max)) - 1
            num_enc = z_in.shape[1]

            slice_pe = positional_encoding(slice_pe.to(z_in.device), int(num_enc/2), include_input = False, pe_freq = self.pe_freq)
            slice_pe = slice_pe.reshape(z_in.shape[0], -1)

            z_in = z_in + slice_pe

        query = self.query_lin(x_in)
        key = self.key_lin(z_in)
        value = self.value_lin(z_in)

        # let z axis attent to slice features
        x_att, att_weights = self.attention(query, key, value, need_weights = True)
        x_att = self.layer_norm1(x_att.squeeze()+x_in)

        x = self.feat_lin1(x_att)
        x = self.activation(x)
        x = self.feat_lin2(x)

        x = self.layer_norm2(x + x_in)
        x = self.activation(x)

        return x, att_weights

class Encoder_block(torch.nn.Module):
    def __init__(self, in_feat = 1, out_feat = 32, layer_norm = False, batch_norm = False, res = None):
        super(Encoder_block, self).__init__()
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_feat,out_feat,3,padding="same")
        self.conv2 = nn.Conv2d(out_feat,out_feat ,3,padding="same")

        if batch_norm == True:
            self.layer1 = nn.BatchNorm2d(out_feat)
            self.layer2 = nn.BatchNorm2d(out_feat)

        if layer_norm == True:
            self.layer1 = torch.nn.LayerNorm((out_feat, res, res))
            self.layer2 = torch.nn.LayerNorm((out_feat, res, res))

    def forward(self, x):
        x = self.conv1(x)
        if self.layer_norm == True or self.batch_norm == True:
            x = self.layer1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.layer_norm == True or self.batch_norm == True:
            x = self.layer2(x)

        return x

class Encoder(torch.nn.Module):
    def __init__(self,
                 num_first = 64,
                 num_feat = 64,   # features of first Conv in block
                 num_feat_out = 64,     # number of global features returned
                 num_feat_attention = 64,
                 num_blocks = 5,
                 in_channels = 1,
                 image_resolution = 128,
                 dropout = 0.,
                 layer_norm = False,
                 batch_norm = False,
                 num_feat_out_xy = 16,
                 spatial_feat = True,
                 global_feat = False,
                 pe_freq = "2pi",
                 get_weights = False
                 ):
        super(Encoder, self).__init__()
        
        # set parameters
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm 
        self.num_feat = num_feat
        self.global_feat = global_feat
        self.spatial_feat = spatial_feat
        self.pe_freq = pe_freq
        self.get_weights = get_weights

        # set up lists for layers
        self.enc_blocks = torch.nn.ModuleList() # Conv Blocks
        self.conv_skip = torch.nn.ModuleList()  # 1x1 Convs for Skip Connection
        self.conv_z = torch.nn.ModuleList()     # 1x1 Convs for slice features
        self.conv_volume = torch.nn.ModuleList()# 1x1 Convs for xy features

        for i in range(num_blocks):
            self.enc_blocks.append(Encoder_block(in_feat = in_channels, out_feat = num_feat, layer_norm = layer_norm, batch_norm = batch_norm, res = image_resolution))
            self.conv_skip.append(nn.Conv2d(in_channels, num_feat, 1))
            self.conv_z.append(nn.Conv2d(num_feat, 32, 1))
            self.conv_volume.append(nn.Conv2d(num_feat, num_feat_out_xy, 1))
            
            # double number of output features after every conv block and set input channels to previous number of features
            # except for the last block
            if i == num_blocks-1:
                None
            else:
                num_feat = int(num_feat *2)
                in_channels = int(num_feat / 2)
                
            # adjust resolution of conv blocks for layer norm 
            image_resolution = int(image_resolution/2) 
            
        # pooling after each Conv Block        
        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        ########################### SELF ATTENTION BLOCKS ###########################################
        num_feat_slices = num_blocks * 32

        self.self_att1 = Self_attention_block(num_feat_slices, num_feat_attention, pos_enc = True, dropout = 0.1, pe_freq = pe_freq)
        self.self_att2 = Self_attention_block(num_feat_slices, num_feat_attention, dropout = 0.1, pe_freq = pe_freq)
        self.self_att3 = Self_attention_block(num_feat_slices, num_feat_attention, dropout = 0.1, pe_freq = pe_freq)#, num_feat_out, extension = extension)

        self.feat_out = torch.nn.Linear(num_feat_slices, num_feat_out)

        ########################### ATTENTION BLOCKS ###########################################
        self.att_blocks_weights = torch.nn.ModuleList()   
        for i in range(int(num_feat_out/32)):
            self.att_blocks_weights.append(Attention_block(32, dropout = 0., pe_freq = pe_freq))
        self.reduce_xy = nn.Linear(num_feat_out_xy * num_blocks, num_feat_out_xy)


    def forward(self, data, slice_index = None, slice_max = None):
        x_in, z_in = data

        # store feature maps for attention and interpolation path
        z_slices = []
        z_volumes = []

        ########################### CONV BLOCKS ###########################################
        for block, skip, conv_z, conv_volume in zip(self.enc_blocks, self.conv_skip, self.conv_z, self.conv_volume):
            z_tmp = skip(z_in)
            z_in = block(z_in)
            z_in = F.relu(z_in+z_tmp)

            # save features maps - pool height and width away for slice features
            z_slices.append(F.relu(F.avg_pool2d(conv_z(z_in), kernel_size = (z_in.shape[2], z_in.shape[3])).squeeze()))
            z_volumes.append(F.relu(conv_volume(z_in)))

            z_in = self.max_pool(z_in)

        z_enc = torch.cat(z_slices, dim = 1)

        ########################### SELF ATTENTION BLOCKS ###########################################
        z_enc_tmp = self.self_att1(z_enc, slice_index, slice_max)
        z_enc_tmp = self.self_att2(z_enc_tmp, slice_index, slice_max)
        z_enc_tmp = self.self_att3(z_enc_tmp, slice_index, slice_max)
        z_enc = z_enc_tmp + z_enc

        z_enc = self.feat_out(z_enc)
        z_enc = F.relu(z_enc)
        
        z_mean = torch.mean(z_enc, axis = 0).squeeze() # for global model
        
        ########################### ATTENTION BLOCKS ###########################################
        # positional encoding for z coordinate of input point
        xz = positional_encoding(x_in[:,0].unsqueeze(-1), num_encoding_functions = 16, include_input=False, pe_freq = self.pe_freq)
        xz = xz.reshape(x_in.shape[0], -1)

        # save slice features and attention weights for each attention block
        x_att = []
        x_weights = []

        for i, att_block in enumerate(self.att_blocks_weights):
            att_feat, att_weights = att_block([z_enc[:, i*32 : (i+1)*32], xz], slice_index = slice_index, slice_max = slice_max)
            x_weights.append(att_weights)
            x_att.append(att_feat)
        
        # attention weights
        slice_weights = torch.stack(x_weights,1) 
        slice_weights = torch.mean(slice_weights, axis = 1)
        
        # slice features
        z_slice = torch.cat(x_att,1) 

        if self.spatial_feat == True:
            ############## XY FEATURES #############
            z_xy_list = []

            grid = x_in[:,1:].reshape(-1,1,1,2)
            grid[:,:,:,0] = grid[:,:,:,0] * -1

            for feat_vol in z_volumes:
                z_xy = torch.einsum("ij, jklm->iklm", slice_weights, feat_vol)

                # interpolate grid points from z
                z_xy = grid_sample(z_xy, grid, mode = "bilinear", align_corners = False).squeeze()      # align_corners = False: extreme values (-1,1) refering to the corner points of input pixel
                z_xy_list.append(z_xy)

            try:
                z_xy = torch.cat(z_xy_list, dim = 1)
            except:
                z_xy = torch.cat(z_xy_list, dim = 0).unsqueeze(0)
            z_xy = self.reduce_xy(z_xy)

            z_xy = F.relu(z_xy)

            z_out = torch.cat((z_slice, z_xy), dim = 1)
            ###########################################

        elif self.global_feat == True:
            z_out = torch.tile(z_mean, (x_in.shape[0], 1))
        
        else: # slice only model
            z_out = z_slice

        z_out = F.relu(z_out)
        z_out = F.dropout(z_out, p = self.dropout)

        if self.get_weights == True:
            return z_out, slice_weights
        else:
            return z_out

class Decoder(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = True, 
                 num_feat_out = 64,
                 num_feat_out_xy = 16,
                 num_encoding_functions = 6,
                 skips = False,
                 dropout = 0.,
                 spatial_feat = False,
                 global_feat = False,
                 pe_freq = "2pi",
                 **_):
        super(Decoder, self).__init__()

        # Set Parameters
        self.spatial_feat = spatial_feat
        self.num_layer = num_layer
        self.pos_enc = pos_encoding
        self.num_encoding_functions = num_encoding_functions
        self.dropout = dropout 
        pos_dim = num_encoding_functions*2*3    # cos + tan for each axis (x,y,z) 
        self.skips = skips
        self.global_feat = global_feat
        self.pe_freq = pe_freq
        
        ########################### OCCUPANCY STUFF ###############################
        # empty lists for layers
        self.layer = torch.nn.ModuleList()
        self.layer_norm = torch.nn.ModuleList()

        if spatial_feat == True:
            num_feat_out_total = num_feat_out + num_feat_out_xy
        else:
            num_feat_out_total = num_feat_out

        # define first layer  
        if pos_encoding == True:
            self.layer.append(torch.nn.Linear(pos_dim + num_feat_out_total,num_nodes_first))
            self.layer_norm.append(torch.nn.LayerNorm(num_nodes_first))
        
        else:
            self.layer.append(torch.nn.Linear(3+num_feat_out_total,num_nodes_first))
            self.layer_norm.append(torch.nn.LayerNorm(num_nodes_first))
            
        # define additional layer      
        for i in range(num_layer-1):
            if i == 0:
                if skips:
                    self.layer.append(torch.nn.Linear(num_nodes_first+ num_feat_out_total, num_nodes))                   
                else:
                    self.layer.append(torch.nn.Linear(num_nodes_first, num_nodes))
                self.layer_norm.append(torch.nn.LayerNorm(num_nodes))

            else:
                if skips:
                    self.layer.append(torch.nn.Linear(num_nodes + num_feat_out_total, num_nodes))
                else:
                    self.layer.append(torch.nn.Linear(num_nodes, num_nodes))        
                self.layer_norm.append(torch.nn.LayerNorm(num_nodes))
                    
        # define last layer with one output node
        self.layer.append(torch.nn.Linear(num_nodes, 1))
        self.activation = torch.nn.ReLU()

    def forward(self, data):
        x_in, z_in = data  
        
        # pos. encoding + dimenstion correction
        if self.pos_enc == True:
            x_pe = positional_encoding(x_in, num_encoding_functions = self.num_encoding_functions, include_input=False, pe_freq = self.pe_freq)
            x_pe = x_pe.reshape(x_pe.shape[0], -1)
            x = torch.cat((x_pe, z_in), dim = 1)          
        else:
            x = torch.cat((x_in, z_in), dim = 1)         

        # first layer
        x = self.layer[0](x)
        #x = self.layer_norm[0](x)  # decreases validation IoU
        x = self.activation(x)

        # subsequent layer
        for fc, norm in zip(self.layer[1:-1], self.layer_norm[1:]):
            if self.skips:
                x = torch.cat((x, z_in), dim = 1)          
            x = fc(x)
            x = norm(x)
            x = self.activation(x)
            
        # last layer
        x = self.layer[-1](x)
    
        return x

# Trainable Feature Model
class MLP_with_feat(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = False, 
                 num_feat = 50,
                 num_lungs = 10,
                 num_encoding_functions = 10,
                 skips = True, **_):
        
        super(MLP_with_feat, self).__init__()

        # initialize parameters for lung fingerprint
        self.lungfinger = torch.nn.ParameterList()
        for i in range(num_lungs):
            self.lungfinger.append(torch.nn.parameter.Parameter(torch.randn(num_feat)))

        self.dec = Decoder(num_layer = num_layer, 
                 num_nodes = num_nodes, 
                 num_nodes_first = num_nodes_first, 
                 pos_encoding = pos_encoding, 
                 num_feat = num_feat,
                 num_feat_out = num_feat,
                 num_encoding_functions = num_encoding_functions,
                 spatial_feat = False,
                 skips = skips,
                 dropout = 0.,
                 pe_freq = "2")
        
        self.img_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad = False) 

    def forward(self, data):

        x,z = data  # z is a list with lung ids
        z = self.lungfinger[z]
        z = torch.tile(z,(x.shape[0], 1))
        x = self.dec([x,z])

        return x

# ISAS model
class ISAS(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = True, 
                 num_feat_attention = 64,
                 num_feat = 64,
                 num_feat_out = 64,
                 num_encoding_functions = 6,
                 num_blocks = 5,
                 skips = False,
                 dropout = 0.,
                 spatial_feat = True,
                 global_feat = False,
                 img_resolution = 128,
                 num_feat_out_xy = 16,
                 layer_norm = False,
                 batch_norm = False,
                 pe_freq = "2pi",
                 get_weights = False,
                 **_):

        super(ISAS, self).__init__()
        
        # Preprocessing - Standardization with mean and standard deviation
        self.img_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad = False) 
        self.get_weights = get_weights

        self.enc_global = Encoder(num_feat = num_feat, 
                 num_feat_out = num_feat_out, 
                 num_feat_attention = num_feat_attention,
                 num_blocks = num_blocks,
                 num_feat_out_xy = num_feat_out_xy,
                 global_feat = global_feat,
                 spatial_feat= spatial_feat,
                 dropout = dropout,
                 image_resolution= img_resolution,
                 layer_norm = layer_norm,
                 batch_norm = batch_norm,
                 pe_freq = pe_freq,
                 get_weights = get_weights)

        self.dec = Decoder(num_layer = num_layer, 
                 num_nodes = num_nodes, 
                 num_nodes_first = num_nodes_first, 
                 pos_encoding = pos_encoding, 
                 num_feat_out = num_feat_out,
                 num_feat_out_xy = num_feat_out_xy,
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = dropout,
                 spatial_feat = spatial_feat,
                 global_feat = True, 
                 pe_freq = pe_freq)

    def forward(self, data, slice_max = None, slice_index = None):
        x_in, z_in, _ = data

        # Encoder
        if self.get_weights == True:
            # extract attention weights
            z_out, slice_weights = self.enc_global([x_in,z_in], slice_index = slice_index, slice_max = slice_max)
        else:
            z_out = self.enc_global([x_in,z_in], slice_index = slice_index, slice_max = slice_max)
           
        # Decoder
        x = self.dec([x_in,z_out])

        if self.get_weights == True:
            return x, slice_weights

        return x


class ISAS_enc_only(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = True, 
                 num_feat_attention = 64,
                 num_feat = 64,
                 num_feat_out = 64,
                 num_encoding_functions = 6,
                 num_blocks = 5,
                 skips = False,
                 dropout = 0.,
                 spatial_feat = True,
                 global_feat = False,
                 img_resolution = 128,
                 num_feat_out_xy = 16,
                 layer_norm = False,
                 batch_norm = False,
                 pe_freq = "2pi",
                 get_weights = False,
                 **_):

        super(ISAS_enc_only, self).__init__()
        
        # Preprocessing - Standardization with mean and standard deviation
        self.img_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad = False) 
        self.get_weights = get_weights

        self.enc_global = Encoder(num_feat = num_feat, 
                 num_feat_out = num_feat_out, 
                 num_feat_attention = num_feat_attention,
                 num_blocks = num_blocks,
                 num_feat_out_xy = num_feat_out_xy,
                 global_feat = global_feat,
                 spatial_feat= spatial_feat,
                 dropout = dropout,
                 image_resolution= img_resolution,
                 layer_norm = layer_norm,
                 batch_norm = batch_norm,
                 pe_freq = pe_freq,
                 get_weights = get_weights)

        self.fc = torch.nn.Linear(num_feat_out + num_feat_out_xy, 1)


    def forward(self, data, slice_max = None, slice_index = None):
        x_in, z_in, _ = data

        # Encoder
        if self.get_weights == True:
            # extract attention weights
            z_out, slice_weights = self.enc_global([x_in,z_in], slice_index = slice_index, slice_max = slice_max)
        else:
            z_out = self.enc_global([x_in,z_in], slice_index = slice_index, slice_max = slice_max)
           
        # Decoder
        x = self.fc(z_out)

        if self.get_weights == True:
            return x, slice_weights

        return x

################################################################
########################### UNET ##############################
################################################################

class Encoder_block_unet(torch.nn.Module):
    def __init__(self, in_feat = 1, out_feat = 32, last_relu = True):
        super(Encoder_block_unet, self).__init__()
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
                self.enc_blocks.append(Encoder_block_unet(in_feat = self.in_channels, out_feat = self.feat, last_relu = last_relu))
            else:
                self.enc_blocks.append(Encoder_block_unet(in_feat = self.in_channels, out_feat = self.feat))
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

class FeatExt(torch.nn.Module):
    def __init__(self, feat = 32, num_blocks = 5):
        super(FeatExt, self).__init__() 
        self.enc = Encoder_unet(feat = feat, num_enc_blocks = num_blocks)
        self.dec = Decoder_unet(feat = feat*(2**(num_blocks-1)), last_relu=False)
       
    def forward(self,x):
        x = self.enc(x)
        x = self.dec(x)

        return x

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


################################################################
########################### 3D UNET ############################
################################################################

class Encoder_block_3D(torch.nn.Module):
    def __init__(self, in_feat = 1, out_feat = 32):
        super(Encoder_block_3D, self).__init__()

        self.conv1 = nn.Conv3d(in_feat,out_feat,3,padding="same")
        self.batch1 = nn.BatchNorm3d(out_feat)
        self.conv2 = nn.Conv3d(out_feat,out_feat ,3,padding="same")
        self.batch2 = nn.BatchNorm3d(out_feat)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)   

        return x 

class Encoder_3D(torch.nn.Module):
    def __init__(self, feat = 32, in_channels = 1, num_enc_blocks = 5):
        super(Encoder_3D, self).__init__()     
        self.feat = feat
        self.in_channels = in_channels

        self.enc_blocks = torch.nn.ModuleList()
        for i in range(num_enc_blocks):
            self.enc_blocks.append(Encoder_block_3D(in_feat = self.in_channels, out_feat = self.feat))
            self.in_channels = self.feat
            self.feat = self.in_channels * 2
        self.max_pool = nn.MaxPool3d(kernel_size = 2, stride = 2)

    def forward(self, x):
        res = []
        for block in self.enc_blocks:
            x = block(x)
            res.append(x)
            x = self.max_pool(x)
        return res

class Decoder_block_3D(torch.nn.Module):
    def __init__(self, feat = 1024, last_relu = True):
        self.last_relu = last_relu
        super(Decoder_block_3D, self).__init__() 
        self.upconv = nn.ConvTranspose3d(feat, feat, kernel_size = 2, stride = 2)
        self.conv1 = nn.Conv3d(feat + int(feat/2), int(feat/2), 3,padding="same")
        feat = int(feat/2)
        self.batch1 = nn.BatchNorm3d(feat)
        self.conv2 = nn.Conv3d(feat, feat, 3,padding="same")
        self.batch2 = nn.BatchNorm3d(feat)

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

class Decoder_3D(torch.nn.Module):
    def __init__(self, feat = 512, num_dec_blocks = 4, last_relu = True):        
        super(Decoder_3D, self).__init__() 
        self.last_relu = last_relu
        self.feat = feat
        self.dec_blocks = torch.nn.ModuleList()

        for i in range(num_dec_blocks):
            if i == num_dec_blocks-1:
                if last_relu == False:
                    self.dec_blocks.append(Decoder_block_3D(feat = self.feat, last_relu = last_relu))
                else:
                    self.dec_blocks.append(Decoder_block_3D(feat = self.feat))
            else:
                self.dec_blocks.append(Decoder_block_3D(feat = self.feat))
            self.feat = int(self.feat/2)


    def forward(self, x):
            for i, block in enumerate(self.dec_blocks):
                x_tmp = block([x[-(i+1)], x[-(i+2)]])
                x[-(i+2)] = x_tmp
            return x[0]

class UNet_3D(torch.nn.Module):
    def __init__(self, feat = 32, num_blocks = 5, **_):
        super(UNet_3D, self).__init__() 
        self.img_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad = False) 

        self.enc = Encoder_3D(feat = feat, num_enc_blocks = num_blocks)
        self.dec = Decoder_3D(feat = feat*(2** (num_blocks-1)), num_dec_blocks = num_blocks-1)

        self.conv_last = nn.Conv3d(feat, 1, 1,padding="same")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.enc(x)
        x = self.dec(x)
        x = F.relu(x)
        x = self.conv_last(x)
        return x