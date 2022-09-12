# load utils
from models import *
from helpers import *
from mesh_render import *
from external_models import *
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
        self.attention = torch.nn.MultiheadAttention(num_feat, 8, dropout = 0.1)

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
    def __init__(self, in_feat = 1, out_feat = 32, last_relu = True, batch_norm = False, res = None):
        super(Encoder_block, self).__init__()
        self.batch_norm = batch_norm
        self.last_relu = last_relu
        self.conv1 = nn.Conv2d(in_feat,out_feat,3,padding="same")
        #self.batch1 = nn.BatchNorm2d(out_feat)
        self.layer1 = torch.nn.LayerNorm((out_feat, res, res))
        self.conv2 = nn.Conv2d(out_feat,out_feat ,3,padding="same")
        #self.batch2 = nn.BatchNorm2d(out_feat)
        self.layer2 = torch.nn.LayerNorm((out_feat, res, res))

    def forward(self, x):
        x = self.conv1(x)
        if self.batch_norm == True:
            x = self.layer1(x)
        x = F.relu(x)
        x = self.conv2(x)
        if self.batch_norm == True:
            x = self.layer2(x)
        # if self.last_relu == True:
        #     x = F.relu(x)  

        return x

class EncoderGlobal(torch.nn.Module):
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
                 masked_attention = False,
                 keep_spatial = False,
                 batch_norm = False,
                 num_feat_out_xy = 16,
                 ):
        super(EncoderGlobal, self).__init__()
        self.dropout = dropout
        self.masked_attention = masked_attention
        self.batch_norm = batch_norm
        self.num_feat = num_feat

        # stem
        self.conv1 = nn.Conv2d(in_channels, num_first, kernel_size = 7, stride = 2, padding = 3)
        #in_channels = num_first
        #image_resolution = int(image_resolution/2)
        self.layer1 = torch.nn.LayerNorm((in_channels, image_resolution, image_resolution))


        # make conv layers
        self.enc_blocks = torch.nn.ModuleList()
        self.conv_skip = torch.nn.ModuleList()
        self.conv_z = torch.nn.ModuleList()
        self.conv_volume = torch.nn.ModuleList()

        for i in range(num_blocks):
            if i == num_blocks-1:
                self.enc_blocks.append(Encoder_block(in_feat = in_channels, out_feat = num_feat, batch_norm = batch_norm, last_relu = last_relu, res = image_resolution))
                self.conv_skip.append(nn.Conv2d(in_channels, num_feat, 1))
                self.conv_z.append(nn.Conv2d(num_feat, 16, 1))
                self.conv_volume.append(nn.Conv2d(num_feat, num_feat_out_xy, 1))

            else:
                self.enc_blocks.append(Encoder_block(in_feat = in_channels, out_feat = num_feat, batch_norm = batch_norm, res = image_resolution))
                self.conv_skip.append(nn.Conv2d(in_channels, num_feat, 1))
                self.conv_z.append(nn.Conv2d(num_feat, 16, 1))
                self.conv_volume.append(nn.Conv2d(num_feat, num_feat_out_xy, 1))

                in_channels = num_feat
            num_feat = in_channels * 2
            image_resolution = int(image_resolution/2)

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # xy features
        self.conv_xy = nn.Conv2d(self.num_feat, num_feat_out_xy, 3, padding="same")

        # reduce channels
        self.conv_reduction = torch.nn.Conv2d(num_feat, num_feat_out, 1)

        num_feat_slices = num_blocks * 16
        self.self_att1 = Self_attention_block(num_feat_slices, num_feat_attention, pos_enc = True, dropout = dropout)
        self.self_att2 = Self_attention_block(num_feat_slices, num_feat_attention, dropout = dropout)
        self.self_att3 = Self_attention_block(num_feat_slices, num_feat_attention, dropout = dropout)#, num_feat_out, extension = extension)

        self.feat_out = torch.nn.Linear(num_feat_slices, num_feat_out)

    def forward(self,z_in, slice_index = None, slice_max = None):

        # STEM
        # z_in = self.conv1(z_in)
        # z_in = self.layer1(z_in)
        # z_in = F.relu(z_in)

        c = 0

        z_slices = []
        z_volumes = []

        # Convolutional Encoder
        for block, skip, conv_z, conv_volume in zip(self.enc_blocks, self.conv_skip, self.conv_z, self.conv_volume):
            z_tmp = skip(z_in)
            z_in = block(z_in)

            # save xy information
            #if c == 0:
                #z_xy = z_in
                #z_xy = torch.mean(z_xy, axis = 0).squeeze()
            z_xy = conv_volume(z_in)
            z_xy = F.dropout(z_xy, p = self.dropout)
            z_xy = F.relu(z_xy)
            z_volumes.append(z_xy)

            z_in = F.relu(z_in+z_tmp)

            # save slice information
            z_tmp = conv_z(z_in)
            z_slices.append(F.relu(F.avg_pool2d(z_tmp, kernel_size = (z_tmp.shape[2], z_tmp.shape[3])).squeeze()))

            z_in = self.max_pool(z_in)
            z_in = F.dropout(z_in, p = self.dropout)
            c += 1

        # pool away spatial structure
        # z_enc = F.avg_pool2d(z_in, kernel_size = (z_in.shape[2], z_in.shape[3])).squeeze()
        # z_enc = F.relu(z_enc)
        z_enc = torch.cat(z_slices, dim = 1)


        # self attention layer
        # z_enc_tmp = self.self_att1(z_enc, slice_index, slice_max)
        # z_enc_tmp = self.self_att2(z_enc_tmp, slice_index, slice_max)
        # z_enc_tmp = self.self_att3(z_enc_tmp, slice_index, slice_max)
        # z_enc = z_enc_tmp + z_enc

        z_enc = self.feat_out(z_enc)

        if self.masked_attention == False:
            # average pooling over slices
            z_enc = torch.mean(z_enc, axis = 0).squeeze()

        z_enc = F.dropout(z_enc, p = self.dropout)
        z_enc = F.relu(z_enc)
        return z_enc, z_volumes

class EncoderLocal(torch.nn.Module):
    def __init__(self,
                 num_feat = 64,   # features of first Conv in block
                 num_blocks = 5,
                 gated_ext = False,
                 img_resolution = 128
                 ):
        super(EncoderLocal, self).__init__()
        if gated_ext == True:
            self.feat_ext = gated(img_size = img_resolution, 
                imgchan = 1, groups = 8, width_per_group = 64)
        else:
            self.feat_ext = FeatExt(feat = num_feat, 
                num_blocks=num_blocks)

    def forward(self, data, resolution = 128, z_resolution = None):
        if z_resolution == None:
            z_resolution = resolution

        x_in, z_in, filter_in = data
        x_in = x_in.squeeze()
        filter_in = filter_in.squeeze()

        z = self.feat_ext(z_in)

        #(slices,feats,h,w) -> (feats,h,w,slices)
        z = z.moveaxis(0,-1)
        
        ############## TRY INTERPOLATE FEATURES #############
        # bring feature volume in right dimension
        z = z.unsqueeze(0)

        # make grid of points to interpolate
        x_tmp,y_tmp,z_tmp = torch.linspace(-1,1,resolution), torch.linspace(-1,1,resolution), torch.linspace(-1,1, z_resolution)
        x_tmp,y_tmp,z_tmp = torch.meshgrid(x_tmp,y_tmp,z_tmp, indexing = "ij")
        grid = torch.stack((z_tmp,y_tmp,x_tmp),3)
        grid = grid.unsqueeze(0).to(x_in.device)
    
        # interpolate grid points from z
        z = grid_sample(z, grid, align_corners = True).squeeze()

        # make feature vector for each coordinate pair
        z = z.reshape(z.shape[0], -1)[:,filter_in]
        z = z.moveaxis(0,-1)

        return z

class Decoder(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = True, 
                 num_feat = 32,
                 num_feat_out = 64,
                 num_feat_out_xy = 32,
                 siren = False, 
                 num_encoding_functions = 6,
                 skips = False,
                 dropout = 0.,
                 spatial_feat = False,
                 masked_attention = False,
                 global_local = False,
                 keep_spatial = False,
                 use_weights = False,
                 num_blocks = 5,
                 **_):
        super(Decoder, self).__init__()

        # variables
        self.spatial_feat = spatial_feat
        self.num_layer = num_layer
        self.pos_enc = pos_encoding
        self.siren = siren
        self.num_encoding_functions = num_encoding_functions
        self.dropout = dropout 
        pos_dim = num_encoding_functions*2*3    # cos + tan for each axis (x,y,z) 
        self.skips = skips
        self.masked_attention = masked_attention
        self.global_local = global_local
        self.keep_spatial = keep_spatial
        self.use_weights = use_weights
        
        ########################### ATTENTION STUFF ###########################################
        if masked_attention == True:
            self.att_blocks = torch.nn.ModuleList()
            for i in range(int(num_feat_out/32)):
                self.att_blocks.append(Attention_block(32, dropout = dropout))

        self.reduce_xy = nn.Linear(num_feat_out_xy * num_blocks, num_feat_out_xy)

        ########################### OCCUPANCY STUFF ###############################
        # compute weight vector for coordinate
        self.feat_weights = nn.Linear(pos_dim, num_feat_out)

        # empty lists for layers
        self.layer = torch.nn.ModuleList()
        self.layer_norm = torch.nn.ModuleList()

        # define first layer 
        if pos_encoding == True:
            if global_local == True:
                num_feat_out_total = num_feat_out + num_feat_out_xy# + num_feat_out
                self.layer.append(torch.nn.Linear(pos_dim + num_feat_out_total,num_nodes_first))
                self.layer_norm.append(torch.nn.LayerNorm(num_nodes_first))

            else:
                num_feat_out_total = num_feat_out 
                self.layer.append(torch.nn.Linear(pos_dim + num_feat_out_total ,num_nodes_first))
                self.layer_norm.append(torch.nn.LayerNorm(num_nodes_first))
        
        else:
            self.layer.append(torch.nn.Linear(3+pos_dim,num_nodes_first))
            self.layer_norm.append(torch.nn.LayerNorm(num_nodes_first))
            
        # define num_layer additional layer      
        for i in range(num_layer-1):
            if i == 0:
                if skips:
                    self.layer.append(torch.nn.Linear(num_nodes_first+ num_feat_out_total, num_nodes))
                    self.layer_norm.append(torch.nn.LayerNorm(num_nodes))
                   
                else:
                    self.layer.append(torch.nn.Linear(num_nodes_first, num_nodes))
                    self.layer_norm.append(torch.nn.LayerNorm(num_nodes))

            else:
                if skips:
                    self.layer.append(torch.nn.Linear(num_nodes + num_feat_out_total, num_nodes))
                    self.layer_norm.append(torch.nn.LayerNorm(num_nodes))

                else:
                    self.layer.append(torch.nn.Linear(num_nodes, num_nodes))        
                    self.layer_norm.append(torch.nn.LayerNorm(num_nodes))
                    
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

    def forward(self, data, resolution, slice_index, slice_max):
        # data
        x_in, z_in = data  # x are the coordinates, z are the ct images

        # pos. encoding + dimenstion correction
        if self.pos_enc == True:
            x = positional_encoding(x_in, num_encoding_functions = self.num_encoding_functions, include_input=False)
            x = x.reshape(x.shape[0], -1)
        elif self.siren == True:
            x = x_in

        if self.global_local == True:
             z_in, z_volume, filter_in = z_in  


        # processing global features
        if self.spatial_feat == False or self.global_local == True:
            if self.masked_attention == False:
                if self.use_weights == True:
                    w = self.feat_weights(x)
                    w = torch.tanh(w)
                    z_in = torch.tile(z_in, (x_in.shape[0],1))
                    z_in = z_in * w

                elif self.global_local == True:
                    None

                else:
                    z_in = torch.tile(z_in, (x_in.shape[0],1))


            elif self.keep_spatial == True:
                None

            else:
                # attention layer
                xz = positional_encoding(x_in[:,0].unsqueeze(-1), num_encoding_functions = 16, include_input=False)
                xz = xz.reshape(x_in.shape[0], -1)

                x_att = []
                x_weights = []
                for i, att_block in enumerate(self.att_blocks):
                    att_feat, att_weights = att_block([z_in[:, i*32 : (i+1)*32], xz], slice_index = slice_index, slice_max = slice_max)
                    x_att.append(att_feat)
                    x_weights.append(att_weights)

                z_slice = torch.cat(x_att,1) 
                slice_weights = torch.stack(x_weights,1) 
                slice_weights = torch.mean(slice_weights, axis = 1)

                z_mean = torch.mean(z_in, axis = 0).squeeze() 
                z_in = z_mean + z_slice


                ############## XY FEATURES #############
                z_xy_list = []

                grid = x_in[:,1:].reshape(-1,1,1,2)
                grid[:,:,:,0] = grid[:,:,:,0] * -1

                #z_xy = z_xy.unsqueeze(0)
                for feat_vol in z_volume:
                    z_xy = torch.einsum("ij, jklm->iklm", slice_weights, feat_vol)

                    # interpolate grid points from z
                    z_xy = grid_sample(z_xy, grid, mode = "bicubic", align_corners = False).squeeze()
                    z_xy_list.append(z_xy)

                z_xy = torch.cat(z_xy_list, dim = 1)
                z_xy = self.reduce_xy(z_xy)
                z_xy = F.relu(z_xy)
                #z_xy = torch.tile(z_xy, (1, z_resolution)).reshape(z_resolution*z_xy.shape[0], z_xy.shape[1])
                #z_xy = z_xy[filter_in]
                ###########################################

        if self.global_local == True:
            z_in = torch.cat((z_slice, z_xy), dim = 1)

        z_in = F.relu(z_in)
        x = torch.cat((x, z_in), dim = 1)          

        x = self.layer[0](x)
        x = self.activation0(x)

        for i in zip(self.layer[1:-1], self.layer_norm[1:]):
            if self.skips:
                x = torch.cat((x,z_in), dim = 1)
            x = i[0](x)
            x = i[1](x)#, z_mean)
            x = self.activation(x)
        x = self.layer[-1](x)
    
        return x

class GlobalNet(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = True, 
                 num_feat_attention = 64,
                 num_feat = 64,
                 num_feat_out = 64,
                 siren = False, 
                 num_encoding_functions = 6,
                 num_blocks = 5,
                 skips = False,
                 dropout = 0.,  
                 masked_attention = False,
                 use_weights = False,
                 batch_norm = False,
                 num_feat_out_xy = 16,
                 **_):

        super(GlobalNet, self).__init__()
        
        self.img_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad = False) 
        # self.img_mean = 0.
        # self.img_std = 1.

        self.enc = EncoderGlobal(num_feat = num_feat, 
                 num_feat_out = num_feat_out, 
                 num_feat_attention = num_feat_attention,
                 num_blocks = num_blocks,
                 masked_attention = masked_attention,
                 batch_norm = batch_norm,
                 num_feat_out_xy = num_feat_out_xy)

        self.dec = Decoder(num_layer = num_layer, 
                 num_nodes = num_nodes, 
                 num_nodes_first = num_nodes_first, 
                 pos_encoding = pos_encoding, 
                 num_feat_out = num_feat_out,
                 siren = siren, 
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = dropout,
                 masked_attention = masked_attention,
                 use_weights = use_weights)

    def forward(self, data, slice_index = None, slice_max = None):

        x_in, z_in = data
        z, _ = self.enc(z_in, slice_index, slice_max)
        x = self.dec([x_in,z], slice_index, slice_max)

        return x


class GlobalVAE(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = True, 
                 num_feat_attention = 64,
                 num_feat = 64,
                 num_feat_out = 64,
                 latent_dim = 64,
                 siren = False, 
                 num_encoding_functions = 6,
                 num_blocks = 5,
                 skips = False,
                 dropout = 0.,
                 masked_attention = False,
                 use_weights = False,
                 **_):

        super(GlobalVAE, self).__init__()
        
        self.img_mean = nn.parameter.Parameter(torch.Tensor([0.]), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.Tensor([1.]), requires_grad = False) 

        self.enc = EncoderGlobal(num_feat = num_feat, 
                 num_feat_out = num_feat_out, 
                 num_feat_attention = num_feat_attention,
                 num_blocks = num_blocks,
                 masked_attention = masked_attention)

        self.mu = torch.nn.Linear(num_feat_out, latent_dim)
        self.std = torch.nn.Linear(num_feat_out, latent_dim)

        self.dec = Decoder(num_layer = num_layer, 
                 num_nodes = num_nodes, 
                 num_nodes_first = num_nodes_first, 
                 pos_encoding = pos_encoding, 
                 num_feat_out = latent_dim,
                 siren = siren, 
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = dropout,
                 masked_attention = masked_attention,
                 use_weights = use_weights)


    def forward(self, data, slice_max = None, slice_index = None):
        x_in, z_in = data
        z = self.enc(z_in)

        mu = self.mu(z)
        logsigma = self.std(z)
        eps = torch.randn((mu.shape)).to(z.device)
        z = mu + (eps * torch.exp(logsigma))
        x = self.dec([x_in,z], slice_index = slice_index, slice_max = slice_max)

        return x, mu, logsigma

class LocalNet(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = True, 
                 num_feat = 64,
                 num_feat_out = 64,
                 siren = False, 
                 num_encoding_functions = 6,
                 num_blocks = 5,
                 skips = False,
                 num_feat_att_marriage = 64,
                 dropout = 0.,
                 gated_ext = False,
                 img_resolution = 128,
                 **_):

        super(LocalNet, self).__init__()
        
        self.img_mean = nn.parameter.Parameter(torch.Tensor([0.]), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.Tensor([1.]), requires_grad = False) 

        if gated_ext == True:
            self.feat_ext = MedT(img_size = img_resolution, 
                imgchan = 1, groups = 4, width_per_group = 64)
        else:
            self.feat_ext = FeatExt(feat = num_feat, 
                num_blocks=num_blocks)

        self.dec = Decoder(num_layer = num_layer, 
                 num_nodes = num_nodes, 
                 num_nodes_first = num_nodes_first, 
                 pos_encoding = pos_encoding, 
                 num_feat_out = num_feat_out,
                 siren = siren, 
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = dropout,
                 spatial_feat = True)

    def forward(self, data, resolution = 128, z_resolution = None):
        if z_resolution == None:
            z_resolution = resolution

        x_in, z_in, filter_in = data
        x_in = x_in.squeeze()
        filter_in = filter_in.squeeze()

        z = self.feat_ext(z_in)
        
        #(slices,feats,h,w) -> (feats,h,w,slices)
        z = z.moveaxis(0,-1)
        
        ############## INTERPOLATE FEATURES #############
        # bring feature volume in right dimension
        z = z.unsqueeze(0)

        # make grid of points to interpolate
        x_tmp,y_tmp,z_tmp = torch.linspace(-1,1,resolution), torch.linspace(-1,1,resolution), torch.linspace(-1,1, z_resolution)
        x_tmp,y_tmp,z_tmp = torch.meshgrid(x_tmp,y_tmp,z_tmp, indexing = "ij")
        grid = torch.stack((z_tmp,y_tmp,x_tmp),3)
        grid = grid.unsqueeze(0).to(x_in.device)
    
        # interpolate grid points from z
        z = grid_sample(z, grid, align_corners = True).squeeze()

        # make feature vector for each coordinate pair
        z = z.reshape(z.shape[0], -1)[:,filter_in]
        z = z.moveaxis(0,-1)

        x = self.dec([x_in,z])

        return x


class GloLoNet(torch.nn.Module):
    def __init__(self, num_layer = 5, 
                 num_nodes = 512, 
                 num_nodes_first = 128, 
                 pos_encoding = True, 
                 num_feat_attention = 64,
                 num_feat = 64,
                 num_feat_out = 64,
                 siren = False, 
                 num_encoding_functions = 6,
                 num_blocks = 5,
                 skips = False,
                 dropout = 0.,
                 masked_attention = False,
                 gated_ext = False,
                 img_resolution = 128,
                 **_):

        super(GloLoNet, self).__init__()
        
        self.img_mean = nn.parameter.Parameter(torch.Tensor([0.]), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.Tensor([1.]), requires_grad = False) 

        self.enc_global = EncoderGlobal(num_feat = num_feat, 
                 num_feat_out = num_feat_out, 
                 num_feat_attention = num_feat_attention,
                 num_blocks = num_blocks,
                 masked_attention = masked_attention)

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
                 siren = siren, 
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = dropout,
                 masked_attention = masked_attention,
                 global_local = True)

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
                 siren = False, 
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
                 siren = siren, 
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = 0.,
                 masked_attention = False,
                 global_local = False)
        
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
                 siren = False, 
                 num_encoding_functions = 6,
                 num_blocks = 5,
                 skips = False,
                 dropout = 0.,
                 masked_attention = False,
                 gated_ext = False,
                 img_resolution = 128,
                 num_feat_out_xy = 16,
                 **_):

        super(GlobalXY, self).__init__()
        self.masked_attention = masked_attention
        self.img_mean = nn.parameter.Parameter(torch.tensor(0.), requires_grad = False)
        self.img_std = nn.parameter.Parameter(torch.tensor(1.), requires_grad = False) 

        self.enc_global = EncoderGlobal(num_feat = num_feat, 
                 num_feat_out = num_feat_out, 
                 num_feat_attention = num_feat_attention,
                 num_blocks = num_blocks,
                 masked_attention = masked_attention,
                 num_feat_out_xy = num_feat_out_xy,
                 image_resolution= img_resolution)

        self.dec = Decoder(num_layer = num_layer, 
                 num_nodes = num_nodes, 
                 num_nodes_first = num_nodes_first, 
                 pos_encoding = pos_encoding, 
                 num_feat = num_feat,
                 num_feat_out = num_feat_out,
                 siren = siren, 
                 num_encoding_functions = num_encoding_functions,
                 skips = skips,
                 dropout = dropout,
                 masked_attention = masked_attention,
                 num_feat_out_xy = num_feat_out_xy,
                 num_blocks = num_blocks,
                 global_local = True)

    def forward(self, data, resolution = 128, z_resolution = None, slice_max = None, slice_index = None):
        x_in, z_in, filter_in = data

       ############## GLOBAL FEATURES #############
        z_global, z_xy = self.enc_global(z_in, slice_index = slice_index, slice_max = slice_max)

        if self.masked_attention == False:
            z_global = torch.tile(z_global, (x_in.shape[0],1))   

        #x = self.dec([x_in,[z_global, z_xy, z_global_smoother]],  slice_index = slice_index, slice_max = slice_max)
        x = self.dec([x_in,[z_global, z_xy, filter_in]],  slice_index = slice_index, slice_max = slice_max, resolution = resolution)

        return x