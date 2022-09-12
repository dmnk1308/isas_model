from os import truncate
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import random
from scipy.stats import rankdata
from torch.nn.init import _calculate_correct_fan
from PIL import Image, ImageFont, ImageDraw 
from torch.nn.functional import grid_sample
import torchvision.transforms.functional as TF
import time 

def load_data(lungs, 
    train = True, 
    point_resolution = 64, 
    img_resolution = 128, 
    batch_size = 250, 
    sample_size = 25000,
    return_mask = False, 
    augmentations = False,
    unet = False):

    point_resolution = int(512/point_resolution)
    img_resolution = int(512/img_resolution)


    data_list = []
    img_list = []
    mask_list = []
    mask_sample_list = []

    if train == True:
        print("Loading Training Data: ")
    else:
        print("Loading Validation Data: ")

    for i in tqdm(lungs):
        mask_raw = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()
        if mask_raw.shape[0] > 50:
            mask_raw = resize(mask_raw, 50)
        # mask to sample points
        mask_sample = mask_raw[:,::point_resolution,::point_resolution]        
        # mask for targets
        mask_target = torch.from_numpy(np.where(mask_raw < 100,  0, 1))[:,::point_resolution,::point_resolution]

        img = nib.load("data/nifti/image/case_"+str(i)+".nii.gz").get_fdata()
        # plt.imsave("test_before.png", img[int(img.shape[0]/2)])
        if img.shape[0] > 50:
            img = resize(img, 50)
        #     plt.imsave("test_after.png", img[int(img.shape[0]/2)])
        img = torch.from_numpy(img)
        img_no_aug = img[:,::img_resolution,::img_resolution]
  
        # make dataloader for unaugmented data
        if unet == False:
            if train == True:
                sample_mask = voxel_sample(mask_sample, flatten = False, n_samples = sample_size, verbose = True)
                data = Lung_dataset(mask_target, sample_mask)
                #dl = DataLoader(data, batch_size = batch_size, shuffle = True)
            else:
                sample_mask = voxel_sample(mask_sample, flatten = False, max_size = False, verbose = False, train = train)
                data = Lung_dataset(mask_target, sample_mask, training = False)
                #dl = DataLoader(data, batch_size = 100000, shuffle = False)
            data_list.append(data)

        # add unaugmented data to list
        img_list.append(img_no_aug)
        mask_list.append(mask_target)
        mask_sample_list.append(mask_sample)
        
        
        # augmentations
        if (augmentations == True) and (train == True):
            img = img/255
            mask_raw = torch.from_numpy(mask_raw/255)

            aug = np.array([True, False, False])
            aug[np.random.randint(1, 3)] = True
            #aug = [True, False, True]

            if aug[0]:
                # padding - fixed scale
                img_pad, mask_pad = pad_fix(img,mask_raw)
                img_pad = img_pad[:,::img_resolution,::img_resolution]*255
                mask_pad = mask_pad[:,::point_resolution,::point_resolution]*255
                mask_pad = mask_pad.numpy()
                mask_pad_bi = torch.from_numpy(np.where(mask_pad < 100,  0, 1))
                if unet == False:
                    sample_mask = voxel_sample(mask_pad, flatten = False, n_samples = sample_size, verbose =False)
                    data = Lung_dataset(mask_pad_bi, sample_mask)
                    #dl = DataLoader(data, batch_size = batch_size, shuffle = True)
                    data_list.append(data)
                img_list.append(img_pad)
                mask_list.append(mask_pad_bi)
                mask_sample_list.append(mask_pad)

                

            if aug[1]:
                # padding - rescaled 
                img_pad, mask_pad = pad(img,mask_raw)
                img_pad = img_pad[:,::img_resolution,::img_resolution]*255
                mask_pad = mask_pad[:,::point_resolution,::point_resolution]*255
                mask_pad = mask_pad.numpy()
                mask_pad_bi = torch.from_numpy(np.where(mask_pad < 100,  0, 1))
                if unet == False:
                    sample_mask = voxel_sample(mask_pad, flatten = False, n_samples = sample_size, verbose =False)
                    data = Lung_dataset(mask_pad_bi, sample_mask)
                    #dl = DataLoader(data, batch_size = batch_size, shuffle = True)
                    data_list.append(data)
                img_list.append(img_pad)
                mask_list.append(mask_pad_bi)
                mask_sample_list.append(mask_pad)

            # cropping
            if aug[2]:
                img_crop, mask_crop = crop(img,mask_raw)
                img_crop = img_crop[:,::img_resolution,::img_resolution]*255
                mask_crop = mask_crop[:,::point_resolution,::point_resolution]*255
                mask_crop = mask_crop.numpy()
                mask_crop_bi = torch.from_numpy(np.where(mask_crop < 100,  0, 1))
                if unet == False:
                    sample_mask = voxel_sample(mask_crop, flatten = False, n_samples = sample_size, verbose =False)
                    data = Lung_dataset(mask_crop_bi, sample_mask)
                    #dl = DataLoader(data, batch_size = batch_size, shuffle = True)
                    data_list.append(data)
                img_list.append(img_crop)
                mask_list.append(mask_crop_bi)
                mask_sample_list.append(mask_crop)

    print("Data Loaded.")

    if return_mask == True:
        return data_list, img_list, mask_list
    return data_list, img_list, mask_sample_list

def crop(image, segmentation):

    height, width = [random.randint(448, 480), random.randint(448, 480)]
    size = (512,512)
    
    imgs = []
    segs = []
    
    for i,s in zip(image, segmentation):
        i = TF.to_pil_image(i.type(torch.float32))
        s = TF.to_pil_image(s.type(torch.float32))
        imgs.append(TF.to_tensor(TF.resize(TF.center_crop(i, (height, width)), size = size)))
        segs.append(TF.to_tensor(TF.resize(TF.center_crop(s, (height, width)), size = size)))

    image = torch.cat(imgs)
    segmentation = torch.cat(segs)

    return image, segmentation


def pad(image, segmentation):

    pa = (random.randint(16, 64),random.randint(16, 64),random.randint(16, 64),random.randint(16, 64))
    size = (512,512)
    imgs = []
    segs = []
    
    for i,s in zip(image, segmentation):
        i = TF.to_pil_image(i.type(torch.float32))
        s = TF.to_pil_image(s.type(torch.float32))
        imgs.append(TF.to_tensor(TF.resize(TF.pad(i, padding = pa), size = size)))
        segs.append(TF.to_tensor(TF.resize(TF.pad(s, padding = pa), size = size)))

    image = torch.cat(imgs)
    segmentation = torch.cat(segs)

    return image, segmentation


def pad_fix(image, segmentation):
    # sample top or bottom; left or right
    choices = np.full(4, False)
    choices[np.random.randint(0,2,size = 1)] = True
    choices[np.random.randint(2,4,size = 1)] = True

    pa = np.full(4, 0)
    pa[choices] = np.random.randint(16,64,size = 2) 
    pa = pa[[2,0,3,1]].tolist() # left, top, right, bottom

    # if we pad on top, the padding has to remain 
    # if we pad on the left, the padding has to remain
    top, left = pa[3], pa[2]

    height = 512
    width = 512
    
    imgs = []
    segs = []
    
    for i,s in zip(image, segmentation):
        i = TF.to_pil_image(i.type(torch.float32))
        s = TF.to_pil_image(s.type(torch.float32))
        imgs.append(TF.to_tensor(TF.crop(TF.pad(i, pa), top, left, height, width)))
        segs.append(TF.to_tensor(TF.crop(TF.pad(s, pa), top, left, height, width)))

    image = torch.cat(imgs)
    segmentation = torch.cat(segs)

    return image, segmentation


def voxel_sample(mask_in, flatten = True, n_samples = 1000000, return_mask = False, max_size = True, proportion_sampling = True, proportion = 0.5, verbose = True, train = True):   
    mask_in = np.moveaxis(mask_in,0,-1)

    if train == False:
        final_mask = np.full(mask_in.shape, True)
        return final_mask

    # make voxel for surface and normal points
    mask_surface = mask_in.copy()
    mask_in = np.where(mask_in >= 100, 1, 0)

    if proportion_sampling == True:
        
        # how many surface points in total
        n_surface = int(proportion*n_samples)

        # potential differences
        occ_diff = 0
        unocc_diff = 0

        # surface occupied
        surface_occ = ((mask_surface<255) * (mask_surface>=100))
        surface_occ = np.stack(np.where(surface_occ),1)
        n_surface_occ = int(n_surface/2)
        if n_surface_occ > surface_occ.shape[0]:
            print("Not enough surface points to sample from.")
            occ_diff = n_surface_occ - surface_occ.shape[0]
            n_surface_occ -= (occ_diff+1)
        surface_occ = surface_occ[np.random.choice(np.arange(surface_occ.shape[0]), n_surface_occ, replace = False)].T

        # surface not occupied
        surface_unocc = ((mask_surface<100) * (mask_surface>0))
        surface_unocc = np.stack(np.where(surface_unocc),1)
        n_surface_unocc = int(n_surface/2)

        if n_surface_unocc > surface_unocc.shape[0]:
            print("Not enough surface points to sample from.")
            unocc_diff = n_surface_unocc - surface_unocc.shape[0]
            n_surface_unocc -= unocc_diff
        surface_unocc = surface_unocc[np.random.choice(np.arange(surface_unocc.shape[0]), n_surface_unocc, replace = False)].T

        final_mask = np.full(mask_surface.shape, False)
        final_mask[surface_occ[0], surface_occ[1], surface_occ[2]] = True
        final_mask[surface_unocc[0], surface_unocc[1], surface_unocc[2]] = True

    # make filter mask for only surface points
    else:
        final_mask = ((mask_surface<255) * (mask_surface>0))
        n_surface = np.sum(final_mask)

    # define number of points to sample
    n_samples_occ = int((n_samples-n_surface)/2) + occ_diff
    n_samples_unocc = int((n_samples-n_surface)/2) + unocc_diff

    # n_samples_occ -= np.sum((mask_surface>=100) * (mask_surface<255))
    # n_samples_unocc -= np.sum((mask_surface>0) * (mask_surface<100))

    # invert surface filter mask to sample from other points
    sample_mask = np.invert(final_mask)

    # occupied points
    sample_mask_occ = sample_mask * (mask_in == 1)
    occ = np.stack(np.where(sample_mask_occ),1)
    if n_samples_occ <= occ.shape[0]:
        filter = occ[np.random.choice(np.arange(occ.shape[0]), n_samples_occ, replace = False)].T
        final_mask[filter[0], filter[1], filter[2]] = True

    else:
        diff = n_samples_occ - occ.shape[0]
        if max_size == True:
            n_samples_occ -= diff
            n_samples_unocc -= diff
            if verbose:
                print("Sample Inbalance - Reduced Sample Size to " + str(n_samples - (2*diff)) + ".")
        else:
            n_samples_unocc += diff
            n_samples_occ -= diff
            if verbose:
                print("Sample Inbalance.")
        filter = occ[np.random.choice(np.arange(occ.shape[0]), n_samples_occ, replace = False)].T
        final_mask[filter[0], filter[1], filter[2]] = True

    # unoccupied points
    sample_mask_unocc = sample_mask * (mask_in == 0)
    unocc = np.stack(np.where(sample_mask_unocc),1)
    if n_samples_unocc <= unocc.shape[0]:
        filter = unocc[np.random.choice(np.arange(unocc.shape[0]), n_samples_unocc, replace = False)].T
        final_mask[filter[0], filter[1], filter[2]] = True
    else:
        print("Sample Inbalance - Unoccupied Points.")

    if flatten == False:
        return final_mask

    x_dim, y_dim, z_dim = mask_in.shape

    # make coordinates
    x = np.linspace(-1,1,x_dim)
    y = np.linspace(1,-1,y_dim)
    z = np.linspace(-1,1,z_dim)

    x,y,z = np.meshgrid(x,y,z, indexing = "ij")
        
    coord_voxel = np.stack((z,y,x),3)
    coord = coord_voxel[final_mask,:]
        
    targets = mask_in[final_mask]
    
    if return_mask == True:
        return coord, targets, final_mask
    
    return coord, targets

def resize(voxel, slices = 50):
    '''
    Function to resize numpy voxel to a given depth
    '''
    D, W, H = voxel.shape

    voxel = np.expand_dims(voxel,0)
    voxel = np.expand_dims(voxel,0)
    voxel = torch.from_numpy(voxel).float()

    z,y,x = torch.linspace(-1,1,slices), torch.linspace(-1,1,W), torch.linspace(-1,1,H)
    x,y,z = torch.meshgrid((x,y,z), indexing = "xy")
    grid = torch.stack((x,y,z),3)
    grid = grid.unsqueeze(0)

    res = grid_sample(voxel, grid, align_corners = True)
    res = np.squeeze(res.numpy())
    res = np.moveaxis(res,-1,0)

    return res

def model_to_voxel(model, device = None, img = None, resolution = 128, z_resolution = None, max_batch = 64 ** 3, 
    spatial_feat = False, xy_feat = False, latent_dim = 64, sample_latent = False, keep_spatial = False, slice_index = None, 
    slice_max = None, no_encoder = False, lung = None):

    model.eval()

    if z_resolution == None:
        z_resolution = resolution

    # prepare image
    img = img.unsqueeze(1)
    img = (img.float().to(device)-model.img_mean)/model.img_std

    # make coordinates
    c = torch.linspace(-1,1,resolution)
    d = torch.linspace(1,-1,resolution)
    e = torch.linspace(-1,1,z_resolution)
    x,y,z = torch.meshgrid(c,d,e, indexing = "ij")
    coord = torch.stack((z,y,x),3).to(device)
    coord = coord.reshape(-1,3)

    pred = torch.zeros(resolution*resolution*z_resolution).to(device)

    # make filter mask
    if spatial_feat == True or xy_feat == True:
        filter_mask = np.full((resolution,resolution,z_resolution), True)
        filter_mask = np.array(np.where(filter_mask.reshape(-1))).squeeze()

    # sample from gaussian
    if sample_latent == True:
        z = torch.randn(latent_dim).to(device)

    num_coord = resolution*resolution*z_resolution
    head = 0

    while head < num_coord:
        idx = np.arange(start = head, stop = min(head + max_batch, num_coord))
        sample_subset = coord[idx]
        if spatial_feat == True or xy_feat == True:
            sample_filter = torch.from_numpy(filter_mask[idx]).to(device)
            y_hat = model((sample_subset.float(), img, sample_filter), resolution = resolution, z_resolution = z_resolution)
        # elif vae == True and sample_latent == True:
        #     y_hat = model.dec([sample_subset.float(), z])
        # elif vae == True:
        #     y_hat,_,_ = model((sample_subset.float(), img))
        elif no_encoder == True:
            y_hat = model((sample_subset.float(), torch.tensor(lung).to(device)))
        else:
            y_hat = model((sample_subset.float(), img), slice_index = slice_index, slice_max = slice_max)
        pred[idx] = y_hat.squeeze().detach()

        head += max_batch

    pred = pred.reshape(resolution, resolution, z_resolution)
    return pred


################################################################
# positional encoding
# copy encoder from NeRF Paper
def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True, random_sampling=False, attention = False, sigma = 2**4):
    """Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    
    if random_sampling == True:
        log_sampling = False
    
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling == True:
        frequency_bands = 2 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    
    elif random_sampling == True:
        B = np.random.normal(loc = 0, scale = sigma, size = (int(num_encoding_functions/3), 3))
        frequency_bands = torch.pi * B
    
    elif attention == True:     
        frequency_bands = 2 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    else:
        frequency_bands = torch.linspace(
            torch.pi ** 0.0,
            torch.pi ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    
    for freq in frequency_bands:
        if random_sampling == True:
            for axis in range(3):
                for func in [torch.sin, torch.cos]:
                    encoding.append(func(tensor * freq[axis]))
        
        elif attention == True:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor/freq))

        else:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )

# Siren 
# use siren activation and initialization from paper 
def siren_uniform_(tensor: torch.Tensor, mode: str = 'fan_in', c: float = 6):
    r"""Fills the input `Tensor` with values according to the method
    described in ` Implicit Neural Representations with Periodic Activation
    Functions.` - Sitzmann, Martel et al. (2020), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \sqrt{\frac{6}{\text{fan\_mode}}}
    Also known as Siren initialization.
    Examples:
        >>> w = torch.empty(3, 5)
        >>> siren.init.siren_uniform_(w, mode='fan_in', c=6)
    :param tensor: an n-dimensional `torch.Tensor`
    :type tensor: torch.Tensor
    :param mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing
        ``'fan_in'`` preserves the magnitude of the variance of the weights in
        the forward pass. Choosing ``'fan_out'`` preserves the magnitudes in
        the backwards pass.s
    :type mode: str, optional
    :param c: value used to compute the bound. defaults to 6
    :type c: float, optional
    """
    fan = _calculate_correct_fan(tensor, mode)
    std = 1 / math.sqrt(fan)
    bound = math.sqrt(c) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
    
class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.
        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')

########### DATA SETS #################
class Lung_dataset(Dataset):
    
    def __init__(self, mask, filter_mask, transform = None, training = True):
        mask = mask.moveaxis(0,-1)
        self.filter_mask = np.array(np.where(filter_mask.reshape(-1))).squeeze()
        self.resolution = mask.shape[1]
        self.z_resolution = mask.shape[-1]
        self.training = training

        # make coordinates of voxel
        x_dim, y_dim, z_dim = mask.shape
        x = np.linspace(-1,1,x_dim)
        y = np.linspace(1,-1,y_dim)
        z = np.linspace(-1,1,z_dim)
        x,y,z  = np.meshgrid(x,y,z, indexing = "ij")
        self.coord = np.stack((z,y,x),3)

        self.targets = mask.reshape(-1)

        self.coord = self.coord.reshape(-1,3) 
        self.weights = torch.tensor([torch.sum(mask[filter_mask] == 0) / torch.sum(mask[filter_mask] == 1)])
        self.transform = transform
        
    def __len__(self):
        return len(self.filter_mask)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = self.coord[self.filter_mask[idx]]
        y = self.targets[self.filter_mask[idx]]

        if self.training == True:
            # add random noise
            # Example: img_res 64:
                    # Abstand zwischen Punkten: 2 * (1/64)
                    # Sample zwischen 0 und 1; ziehe 0.5 und teile durch 1/32
            X[:2] = X[:2]+((np.random.random_sample(X[:2].shape)-0.5)/(self.resolution/2))
            X[-1] = X[-1]+((np.random.random_sample(X[-1].shape)-0.5)/(self.z_resolution/2))

            # correct at boundaries
            # X[X < -1] = -1
            # X[X > 1] = 1 

        #sample_idx = np.array([self.data[0][idx],self.data[1][idx],self.data[2][idx]])
        sample_idx = self.filter_mask[idx]
        
        sample = [X, sample_idx, y, self.weights]

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
######################## METRICS #################################
def psnr(img1, img2):
    if torch.is_tensor(img1) == False:
        img1 = torch.from_numpy(img1)
    if torch.is_tensor(img2) == False:
        img2 = torch.from_numpy(img2)
    mse = torch.mean((img1.float() - img2.float()) ** 2)
    return 20 * torch.log10(torch.tensor(255.0)) - 10 * torch.log10(mse)

def iou_loss(t_hat, target):

    # flatten input
    pred_tmp = torch.flatten(t_hat)
    target_tmp = torch.flatten(target)

    intersection = torch.sum(pred_tmp * target_tmp)
    union = torch.sum(pred_tmp) + torch.sum(target_tmp) - torch.sum(pred_tmp * target_tmp)
    iou = torch.div(- intersection, union)
    
    return iou

def iou(t_hat, target):

    # flatten input
    pred_tmp = torch.flatten(t_hat)
    target_tmp = torch.flatten(target)

    intersection = torch.sum(pred_tmp * target_tmp)
    union = torch.sum(pred_tmp) + torch.sum(target_tmp) - torch.sum(pred_tmp * target_tmp)
    iou = torch.div(intersection, union)
    
    return iou
        
def dice_loss(input, target):
    smooth = 1.

    iflat = input.reshape(-1)
    tflat = target.reshape(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

def dice_coef(input, target):
    smooth = 1.

    #iflat = input.view(-1)
    #tflat = target.view(-1)
    intersection = (input * target).sum()
    
    return (2. * intersection + smooth) / (input.sum() + target.sum() + smooth) 

def vae_loss(recon_x, x, mu, logsigma):
    # Reconstruction losses are calculated using Mean Squared Error (MSE) and 
    # summed over all elements and batch
    #rec_loss = torch.sum(-(recon_x - x)**2)
    rec_loss = torch.nn.BCEWithLogitsLoss()(recon_x, x)

    # See Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kl_loss = -(0.5 * torch.sum((1+(2*logsigma) - (mu**2) - (torch.exp(logsigma)**2))))

    total_loss = kl_loss + rec_loss

    return total_loss, rec_loss, kl_loss