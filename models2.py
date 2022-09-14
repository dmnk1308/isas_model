# load utils
from models import *
from helpers import *
from render import *
from archive.external_models import *
#from end2end_helpers import *
#from Resnet import *

class Self_attention_block(torch.nn.Module):
    def __init__(self,
                num_feat_in = 256, 
                num_feat = 32,
                pos_enc = False,
                dropout = 0.1):
        super(Self_attention_block, self).__init__()
        self.num_feat = num_feat
        self.pos_enc = pos_enc

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
            slice_pe = 2*(slice_pe/slice_max) - 1
            num_enc = z_enc_tmp.shape[1]

            slice_pe = positional_encoding(slice_pe.to(z_enc.device), int(num_enc/2), log_sampling = False, attention = True, include_input = False)
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
                dropout = 0.1):
        super(Attention_block, self).__init__()

        self.pos_encoding = pos_encoding

        # define self attention layer
        self.attention = torch.nn.MultiheadAttention(num_feat, 8, dropout = 0.1, batch_first = True)

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
            slice_pe = 2*(slice_pe/slice_max) - 1
            num_enc = z_in.shape[1]

            slice_pe = positional_encoding(slice_pe.to(z_in.device), int(num_enc/2), log_sampling = False, attention = True, include_input = False)
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
    def __init__(self, in_feat = 1, out_feat = 32, last_relu = True, layer_norm = False, res = None):
        super(Encoder_block, self).__init__()
        self.layer_norm = layer_norm
        self.last_relu = last_relu
        self.conv1 = nn.Conv2d(in_feat,out_feat,3,padding="same")
        #self.batch1 = nn.BatchNorm2d(out_feat)
        self.layer1 = torch.nn.LayerNorm((out_feat, res, res))
        self.conv2 = nn.Conv2d(out_feat,out_feat ,3,padding="same")
        #self.batch2 = nn.BatchNorm2d(out_feat)
        self.layer2 = torch.nn.LayerNorm((out_feat, res, res))

    def forward(self, x):
        x = self.conv1(x)
        if self.layer_norm == True:
            x = self.layer1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.layer_norm == True:
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
                 last_relu = True,
                 dropout = 0.,
                 layer_norm = False,
                 num_feat_out_xy = 16,
                 spatial_feat = True,
                 global_feat = False
                 ):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.num_feat = num_feat
        self.global_feat = global_feat
        self.spatial_feat = spatial_feat

        # stem
        self.conv1 = nn.Conv2d(in_channels, num_first, kernel_size = 7, stride = 2, padding = 3)
        in_channels = num_first
        image_resolution = int(image_resolution/2)
        self.layer1 = torch.nn.LayerNorm((in_channels, image_resolution, image_resolution))
        self.conv_z_first = nn.Conv2d(num_feat, 16, 1)
        self.conv_volume_first = nn.Conv2d(num_feat, num_feat_out_xy, 1)

        ########################### RESNET ENCODER ###########################################
        self.enc_blocks = torch.nn.ModuleList()
        self.conv_skip = torch.nn.ModuleList()
        self.conv_z = torch.nn.ModuleList()
        self.conv_volume = torch.nn.ModuleList()

        for i in range(num_blocks):
            if i == num_blocks-1:
                self.enc_blocks.append(Encoder_block(in_feat = in_channels, out_feat = num_feat, layer_norm = layer_norm, last_relu = last_relu, res = image_resolution))
                self.conv_skip.append(nn.Conv2d(in_channels, num_feat, 1))
                self.conv_z.append(nn.Conv2d(num_feat, 16, 1))
                self.conv_volume.append(nn.Conv2d(num_feat, num_feat_out_xy, 1))

            else:
                self.enc_blocks.append(Encoder_block(in_feat = in_channels, out_feat = num_feat, layer_norm = layer_norm, res = image_resolution))
                self.conv_skip.append(nn.Conv2d(in_channels, num_feat, 1))
                self.conv_z.append(nn.Conv2d(num_feat, 16, 1))
                self.conv_volume.append(nn.Conv2d(num_feat, num_feat_out_xy, 1))

                in_channels = num_feat
            num_feat = in_channels * 2

            image_resolution = int(image_resolution/2) # adjust resolution due to stem
        
        num_blocks = num_blocks + 1 # adjust length of feature map lists due to stem

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # xy features
        self.conv_xy = nn.Conv2d(self.num_feat, num_feat_out_xy, 3, padding="same")

        # reduce channels
        self.conv_reduction = torch.nn.Conv2d(num_feat, num_feat_out, 1)

        ########################### SELF ATTENTION ###########################################
        num_feat_slices = num_blocks * 16

        self.self_att1 = Self_attention_block(num_feat_slices, num_feat_attention, pos_enc = True, dropout = 0.1)
        self.self_att2 = Self_attention_block(num_feat_slices, num_feat_attention, dropout = 0.1)
        self.self_att3 = Self_attention_block(num_feat_slices, num_feat_attention, dropout = 0.1)#, num_feat_out, extension = extension)

        self.feat_out = torch.nn.Linear(num_feat_slices, num_feat_out)

        ########################### MASKED ATTENTION  ###########################################
        self.att_blocks_weights = torch.nn.ModuleList()
        for i in range(int(num_feat_out/32)):
            self.att_blocks_weights.append(Attention_block(32, dropout = 0.1))
        self.slice_norm = torch.nn.LayerNorm(num_feat_out)

        # self.att_blocks_feat = torch.nn.ModuleList()
        # for i in range(int(num_feat_out/32)):
        #     self.att_blocks_feat.append(Attention_block(32, dropout = dropout))

        self.reduce_xy = nn.Linear(num_feat_out_xy * num_blocks, num_feat_out_xy)

        self.reduce_xy_norm = torch.nn.LayerNorm(num_feat_out_xy)

        if spatial_feat == True:
            self.reduce_feat = nn.Linear(num_feat_out + num_feat_out_xy, num_feat_out)
            self.reduce_feat_norm = torch.nn.LayerNorm(num_feat_out)


    def forward(self, data, slice_index = None, slice_max = None):

        x_in, z_in = data

        z_slices = []
        z_volumes = []

        # STEM
        z_in = self.conv1(z_in)
        if self.layer_norm == True:
            z_in = self.layer1(z_in)
        z_in = F.relu(z_in)

        # save features maps
        z_slices.append(F.relu(F.avg_pool2d(self.conv_z_first(z_in), kernel_size = (z_in.shape[2], z_in.shape[3])).squeeze()))
        z_volumes.append(F.relu(self.conv_volume_first(z_in)))

        ########################### RESNET ENCODER ###########################################
        for block, skip, conv_z, conv_volume in zip(self.enc_blocks, self.conv_skip, self.conv_z, self.conv_volume):
            z_tmp = skip(z_in)
            z_in = block(z_in)
            z_in = F.relu(z_in+z_tmp)

            # save features maps
            z_slices.append(F.relu(F.avg_pool2d(conv_z(z_in), kernel_size = (z_in.shape[2], z_in.shape[3])).squeeze()))
            z_volumes.append(F.relu(conv_volume(z_in)))

            z_in = self.max_pool(z_in)

        z_enc = torch.cat(z_slices, dim = 1)

        ########################### SELF ATTENTION  ###########################################
        z_enc_tmp = self.self_att1(z_enc, slice_index, slice_max)
        z_enc_tmp = self.self_att2(z_enc_tmp, slice_index, slice_max)
        z_enc_tmp = self.self_att3(z_enc_tmp, slice_index, slice_max)
        z_enc = z_enc_tmp + z_enc

        z_enc = self.feat_out(z_enc)
        z_enc = F.relu(z_enc)

        ########################### MASKED ATTENTION  ###########################################
        xz = positional_encoding(x_in[:,0].unsqueeze(-1), num_encoding_functions = 16, include_input=False)
        xz = xz.reshape(x_in.shape[0], -1)

        x_att = []
        x_weights = []

        for i, att_block in enumerate(self.att_blocks_weights):
            att_feat, att_weights = att_block([z_enc[:, i*32 : (i+1)*32], xz], slice_index = slice_index, slice_max = slice_max)
            x_weights.append(att_weights)
            x_att.append(att_feat)
        slice_weights = torch.stack(x_weights,1) 
        slice_weights = torch.mean(slice_weights, axis = 1)
        z_slice = torch.cat(x_att,1) 
        #if self.layer_norm == True:
        #    z_slice = self.slice_norm(z_slice)
        z_mean = torch.mean(z_enc, axis = 0).squeeze() 
        #z_slice = z_mean + z_slice

        ########################### MASKED ATTENTION FEAT ###########################################
        # for i, att_block in enumerate(self.att_blocks_feat):
        #     att_feat, att_weights = att_block([z_enc[:, i*32 : (i+1)*32], xz], slice_index = slice_index, slice_max = slice_max)
        #     x_att.append(att_feat)

        # z_slice = torch.cat(x_att,1) 
        
        # z_mean = torch.mean(z_in, axis = 0).squeeze() 
        # z_slice = z_mean + z_slice

        if self.spatial_feat == True:
            ############## XY FEATURES #############
            z_xy_list = []

            grid = x_in[:,1:].reshape(-1,1,1,2)
            grid[:,:,:,0] = grid[:,:,:,0] * -1

            for feat_vol in z_volumes:
                z_xy = torch.einsum("ij, jklm->iklm", slice_weights, feat_vol)

                # interpolate grid points from z
                z_xy = grid_sample(z_xy, grid, mode = "bicubic", align_corners = False).squeeze()
                z_xy_list.append(z_xy)

            z_xy = torch.cat(z_xy_list, dim = 1)
            z_xy = self.reduce_xy(z_xy)
            #if self.layer_norm == True:
            #    z_xy = self.reduce_xy_norm(z_xy)
            z_xy = F.relu(z_xy)

            z_out = torch.cat((z_slice, z_xy), dim = 1)
            ###########################################

        elif self.global_feat == True:
            z_out = torch.tile(z_mean, (x_in.shape[0], 1))
        
        else:
            z_out = z_slice

        z_out = F.relu(z_out)
        z_out = F.dropout(z_out, p = self.dropout)

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
                 **_):
        super(Decoder, self).__init__()

        # variables
        self.spatial_feat = spatial_feat
        self.num_layer = num_layer
        self.pos_enc = pos_encoding
        self.num_encoding_functions = num_encoding_functions
        self.dropout = dropout 
        pos_dim = num_encoding_functions*2*3    # cos + tan for each axis (x,y,z) 
        self.skips = skips
        self.global_feat = global_feat
        
        ########################### OCCUPANCY STUFF ###############################
        # empty lists for layers
        self.layer = torch.nn.ModuleList()
        self.layer_norm = torch.nn.ModuleList()

        # define first layer 
        if pos_encoding == True:
            if spatial_feat == True:
                num_feat_out_total = num_feat_out + num_feat_out_xy
            else:
                num_feat_out_total = num_feat_out
            self.layer.append(torch.nn.Linear(pos_dim + num_feat_out_total,num_nodes_first))
            self.layer_norm.append(torch.nn.LayerNorm(num_nodes_first))
        
        else:
            self.layer.append(torch.nn.Linear(3+pos_dim,num_nodes_first))
            self.layer_norm.append(torch.nn.LayerNorm(num_nodes_first))
            
        # define num_layer additional layer      
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

        self.activation0 = torch.nn.ReLU()
        self.activation = torch.nn.ReLU()

    def forward(self, data):
        # data
        x_in, z_in = data  # x are the coordinates, z are the ct images

        # pos. encoding + dimenstion correction
        if self.pos_enc == True:
            x = positional_encoding(x_in, num_encoding_functions = self.num_encoding_functions, include_input=False)
            x = x.reshape(x.shape[0], -1)

        x = torch.cat((x, z_in), dim = 1)          

        x = self.layer[0](x)
        #x = self.layer_norm[0](x)
        x = self.activation0(x)

        for i in zip(self.layer[1:-1], self.layer_norm[1:]):
            if self.skips:
                x = torch.cat((x, z_in), dim = 1)          
            x = i[0](x)
            x = i[1](x)
            x = self.activation(x)
        x = self.layer[-1](x)
    
        return x

class GloLoNet(torch.nn.Module):
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
                 gated_ext = False,
                 img_resolution = 128,
                 **_):

        super(GloLoNet, self).__init__()
        
        self.img_mean = nn.parameter.Parameter(torch.Tensor([0.]), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.Tensor([1.]), requires_grad = False) 

        self.enc_global = Encoder(num_feat = num_feat, 
                 num_feat_out = num_feat_out, 
                 num_feat_attention = num_feat_attention,
                 num_blocks = num_blocks)

        self.enc_local = EncoderLocal(num_feat = num_feat, 
                num_blocks=num_blocks, gated_ext = gated_ext, 
                img_resolution=img_resolution)

        self.local_gate = nn.Linear(num_feat, 1)

        self.dec = Decoder(num_layer = num_layer, 
                 num_nodes = num_nodes, 
                 num_nodes_first = num_nodes_first, 
                 pos_encoding = pos_encoding, 
                 num_feat = num_feat,
                 num_feat_out = num_feat_out,
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = dropout)

    def forward(self, data, resolution = 128, z_resolution = None):
        x_in, z_in, filter_in = data
        z_global = self.enc_global(z_in)
        z_local = self.enc_local([x_in, z_in, filter_in], resolution = resolution, z_resolution = z_resolution)
        local_out = torch.sigmoid(self.local_gate(z_local))
        p = 0.5 * (torch.cos(2*torch.pi*local_out)+1)
        #print(torch.round(p,decimals = 2))
        
        x = self.dec([x_in,[(1-p) * z_global, p * z_local]])

        return x

class MLP_with_feat(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = False, 
                 num_feat = 50,
                 num_lungs = 10,
                 num_encoding_functions = 6,
                 skips = True,
                 num_feat_out_xy = 16, **_):
        
        super(MLP_with_feat, self).__init__()

        # initialize parameters for lung fingerprint
        self.lungfinger = torch.nn.ParameterList()
        for i in range(num_lungs):
            self.lungfinger.append(torch.nn.parameter.Parameter(torch.randn(num_feat)))
            #self.lungfinger.append(torch.nn.parameter.Parameter(torch.zeros(num_feat)))

        self.dec = Decoder(num_layer = num_layer, 
                 num_nodes = num_nodes, 
                 num_nodes_first = num_nodes_first, 
                 pos_encoding = pos_encoding, 
                 num_feat = num_feat,
                 num_feat_out = num_feat,
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = 0.)
        
    def forward(self, data):

        x,z = data  # z is a list with lung ids
        z = self.lungfinger[z]
        #z = torch.tile(x.shape[0], 1)
        x = self.dec([x,z])

        return x


class GlobalXY(torch.nn.Module):
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
                 **_):

        super(GlobalXY, self).__init__()
        self.img_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad = False) 

        self.enc_global = Encoder(num_feat = num_feat, 
                 num_feat_out = num_feat_out, 
                 num_feat_attention = num_feat_attention,
                 num_blocks = num_blocks,
                 num_feat_out_xy = num_feat_out_xy,
                 global_feat = global_feat,
                 spatial_feat= spatial_feat,
                 dropout = dropout,
                 image_resolution= img_resolution,
                 layer_norm = layer_norm)

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
                 global_feat = True)

    def forward(self, data, resolution = 128, z_resolution = None, slice_max = None, slice_index = None):
        x_in, z_in, filter_in = data
       ############## GLOBAL FEATURES #############
        z_out = self.enc_global([x_in,z_in], slice_index = slice_index, slice_max = slice_max)

        x = self.dec([x_in,z_out])

        return x



### UNET ###

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