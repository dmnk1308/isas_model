from os import truncate
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import random
from scipy.stats import rankdata
from torch.nn.init import _calculate_correct_fan
from torch.nn.functional import grid_sample
import torchvision.transforms.functional as TF

def load_data(lungs, 
    train = True, 
    point_resolution = 64, 
    img_resolution = 128, 
    batch_size = 250, 
    sample_size = 25000,
    return_mask = False, 
    augmentations = False,
    unet = False,
    unet3D = False,
    proportion = 1.0):

    point_resolution = int(512/point_resolution)
    img_resolution = int(512/img_resolution)

    dl_list = []
    img_list = []
    mask_list = []

    if train == True:
        print("Loading Training Data: ")
    else:
        print("Loading Validation Data: ")

    for i in tqdm(lungs):
        mask_raw = nib.load("data/nifti/mask/case_"+str(i)+".nii.gz").get_fdata()
        #if unet3D == True:
        mask_raw = resize(mask_raw, 20)
        # elif mask_raw.shape[0] > 50:
        #     mask_raw = resize(mask_raw, 50)
        # mask to sample points
        mask_sample = mask_raw[:,::point_resolution,::point_resolution]        
        # mask for targets
        mask_target = torch.from_numpy(np.where(mask_raw < 100,  0, 1))[:,::point_resolution,::point_resolution]

        img = nib.load("data/nifti/image/case_"+str(i)+".nii.gz").get_fdata()
        #if unet3D == True:
        img = resize(img, 20)
        #elif img.shape[0] > 50:
         #   img = resize(img, 50)
        img = torch.from_numpy(img)
        img_no_aug = img[:,::img_resolution,::img_resolution]
  
        # make dataloader for unaugmented data
        if unet == False:
            if train == True:
                sample_mask = voxel_sample(mask_sample, flatten = False, n_samples = sample_size, verbose = True, proportion = proportion)
                data = Lung_dataset(mask_target, sample_mask)
                dl = DataLoader(data, batch_size = batch_size, shuffle = True)
            else:
                sample_mask = voxel_sample(mask_sample, flatten = False, max_size = False, verbose = False, train = train)
                data = Lung_dataset(mask_target, sample_mask, training = False)
                dl = DataLoader(data, batch_size = batch_size, shuffle = False)
            dl_list.append(dl)

        # add unaugmented data to list
        img_list.append(img_no_aug)
        mask_list.append(mask_target)
        
        
        # augmentations
        if (augmentations == True) and (train == True):
            img = img/255
            mask_raw = torch.from_numpy(mask_raw/255)

            aug = np.array([True, False, False])
            aug[np.random.randint(1, 3)] = True

            if aug[0]:
                # padding - fixed scale
                img_pad, mask_pad = pad_fix(img,mask_raw)
                img_pad = img_pad[:,::img_resolution,::img_resolution]*255
                mask_pad = mask_pad[:,::point_resolution,::point_resolution]*255
                mask_pad = mask_pad.numpy()
                mask_pad_bi = torch.from_numpy(np.where(mask_pad < 100,  0, 1))
                if unet == False:
                    sample_mask = voxel_sample(mask_pad, flatten = False, n_samples = sample_size, verbose =False, proportion = proportion)
                    data = Lung_dataset(mask_pad_bi, sample_mask)
                    dl = DataLoader(data, batch_size = batch_size, shuffle = True)
                    dl_list.append(dl)
                img_list.append(img_pad)
                mask_list.append(mask_pad_bi)
                

            if aug[1]:
                # padding - rescaled -> zoom out
                img_pad, mask_pad = pad(img,mask_raw)
                img_pad = img_pad[:,::img_resolution,::img_resolution]*255
                mask_pad = mask_pad[:,::point_resolution,::point_resolution]*255
                mask_pad = mask_pad.numpy()
                mask_pad_bi = torch.from_numpy(np.where(mask_pad < 100,  0, 1))
                if unet == False:
                    sample_mask = voxel_sample(mask_pad, flatten = False, n_samples = sample_size, verbose =False, proportion = proportion)
                    data = Lung_dataset(mask_pad_bi, sample_mask)
                    dl = DataLoader(data, batch_size = batch_size, shuffle = True)
                    dl_list.append(dl)
                img_list.append(img_pad)
                mask_list.append(mask_pad_bi)

            if aug[2]:
                # cropping -> zoom in
                img_crop, mask_crop = crop(img,mask_raw)
                img_crop = img_crop[:,::img_resolution,::img_resolution]*255
                mask_crop = mask_crop[:,::point_resolution,::point_resolution]*255
                mask_crop = mask_crop.numpy()
                mask_crop_bi = torch.from_numpy(np.where(mask_crop < 100,  0, 1))
                if unet == False:
                    sample_mask = voxel_sample(mask_crop, flatten = False, n_samples = sample_size, verbose =False, proportion = proportion)
                    data = Lung_dataset(mask_crop_bi, sample_mask)
                    dl = DataLoader(data, batch_size = batch_size, shuffle = True)
                    dl_list.append(dl)
                img_list.append(img_crop)
                mask_list.append(mask_crop_bi)

    print("Data Loaded.")

    if return_mask == True:
        return dl_list, img_list, mask_list
    return dl_list, img_list

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


def voxel_sample(mask_in, flatten = True, n_samples = 1000000, random = False, unbalanced = False, return_mask = False, max_size = True, proportion_sampling = True, proportion = 0.5, verbose = True, train = True):   
    mask_in = np.moveaxis(mask_in,0,-1)

    if train == False:
        final_mask = np.full(mask_in.shape, True)
        return final_mask
    
    if random == True:
        final_mask = np.full(mask_in.shape, False)
        
        samples_occ = np.where(mask_in >= 100,1,0)
        samples_occ = np.stack(np.where(samples_occ),1) 
        try:       
            samples_occ = samples_occ[np.random.choice(np.arange(samples_occ.shape[0]), int(n_samples/2), replace = False)].T
        except:
            samples_occ = samples_occ[np.random.choice(np.arange(samples_occ.shape[0]), samples_occ.shape[0], replace = False)].T
            n_samples = int(samples_occ.shape[0]*2)
        final_mask[samples_occ[0], samples_occ[1], samples_occ[2]] = True
        
        samples_unocc = np.where(mask_in < 100,1,0)
        samples_unocc = np.stack(np.where(samples_unocc),1)   
        samples_unocc = samples_unocc[np.random.choice(np.arange(samples_unocc.shape[0]), int(n_samples/2), replace = False)].T
        final_mask[samples_unocc[0], samples_unocc[1], samples_unocc[2]] = True
        return final_mask
    
    if unbalanced == True:
        final_mask = np.full(mask_in.shape, False)
        
        samples = np.where(mask_in >= 0,1,0)
        samples = np.stack(np.where(samples),1)        
        samples = samples[np.random.choice(np.arange(samples.shape[0]), n_samples, replace = False)].T
        final_mask[samples[0], samples[1], samples[2]] = True

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
            #print("Not enough surface points to sample from.")
            occ_diff = n_surface_occ - surface_occ.shape[0]
            n_surface_occ -= occ_diff
            n_samples -= n_surface_occ
        surface_occ = surface_occ[np.random.choice(np.arange(surface_occ.shape[0]), n_surface_occ, replace = False)].T

        # surface not occupied
        surface_unocc = ((mask_surface<100) * (mask_surface>0))
        surface_unocc = np.stack(np.where(surface_unocc),1)
        n_surface_unocc = int(n_surface/2)

        if n_surface_unocc > surface_unocc.shape[0]:
            unocc_diff = n_surface_unocc - surface_unocc.shape[0]
            n_surface_unocc -= unocc_diff
            n_samples -= n_surface_unocc
        surface_unocc = surface_unocc[np.random.choice(np.arange(surface_unocc.shape[0]), n_surface_unocc, replace = False)].T

        final_mask = np.full(mask_surface.shape, False)
        final_mask[surface_occ[0], surface_occ[1], surface_occ[2]] = True
        final_mask[surface_unocc[0], surface_unocc[1], surface_unocc[2]] = True

    # make filter mask for only surface points
    else:
        final_mask = ((mask_surface<255) * (mask_surface>0))
        n_surface = np.sum(final_mask)

    # invert surface filter mask to sample from other points
    sample_mask = np.invert(final_mask)

    # #### RANDOM SAMPLE OTHER POINTS
    occ = np.stack(np.where(sample_mask),1)
    filter = occ[np.random.choice(np.arange(occ.shape[0]), n_samples, replace = False)].T
    final_mask[filter[0], filter[1], filter[2]] = True
    # ########################################################################

    ################################################################
    # # define number of points to sample
    # n_samples_occ = int((n_samples-n_surface)/2) + occ_diff
    # n_samples_unocc = int((n_samples-n_surface)/2) + unocc_diff

    # # occupied points
    # sample_mask_occ = sample_mask * (mask_in == 1)
    # occ = np.stack(np.where(sample_mask_occ),1)
    # if n_samples_occ <= occ.shape[0]:
    #     filter = occ[np.random.choice(np.arange(occ.shape[0]), n_samples_occ, replace = False)].T
    #     final_mask[filter[0], filter[1], filter[2]] = True

    # else:
    #     diff = n_samples_occ - occ.shape[0]
    #     if max_size == True:
    #         n_samples_occ -= diff
    #         n_samples_unocc -= diff
    #         if verbose:
    #             print("Sample Inbalance - Reduced Sample Size to " + str(n_samples - (2*diff)) + ".")
    #     else:
    #         n_samples_unocc += diff
    #         n_samples_occ -= diff
    #         if verbose:
    #             print("Sample Inbalance.")
    #     filter = occ[np.random.choice(np.arange(occ.shape[0]), n_samples_occ, replace = False)].T
    #     final_mask[filter[0], filter[1], filter[2]] = True

    # # unoccupied points
    # sample_mask_unocc = sample_mask * (mask_in == 0)
    # unocc = np.stack(np.where(sample_mask_unocc),1)
    # if n_samples_unocc <= unocc.shape[0]:
    #     filter = unocc[np.random.choice(np.arange(unocc.shape[0]), n_samples_unocc, replace = False)].T
    #     final_mask[filter[0], filter[1], filter[2]] = True
    # else:
    #     print("Sample Inbalance - Unoccupied Points.")
    ################################################################


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
    slice_index = None, slice_max = None, no_encoder = False, lung = None, get_weights = False):

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
    filter_mask = np.full((resolution,resolution,z_resolution), True)
    filter_mask = np.array(np.where(filter_mask.reshape(-1))).squeeze()

    num_coord = resolution*resolution*z_resolution
    head = 0

    att = []

    while head < num_coord:
        idx = np.arange(start = head, stop = min(head + max_batch, num_coord))
        sample_subset = coord[idx]

        if no_encoder == True:
            y_hat = model((sample_subset.float(), torch.tensor(lung).to(device)))

        else:
            sample_filter = torch.from_numpy(filter_mask[idx]).to(device)
            if get_weights == False:
                y_hat = model((sample_subset.float(), img, sample_filter), slice_max = slice_max, slice_index = slice_index)
            else:
                y_hat, weights = model((sample_subset.float(), img, sample_filter), slice_max = slice_max, slice_index = slice_index)
                att.append(weights.detach().cpu().numpy())

        pred[idx] = y_hat.squeeze().detach()
        head += max_batch

    pred = pred.reshape(resolution, resolution, z_resolution)

    if get_weights == True:
        att = np.concatenate(att, axis = 0)
        return pred, att

    return pred


################################################################
# positional encoding
# copy encoder from NeRF Paper
def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True, random_sampling=False, attention = False, sigma = 2**4, freq = 2., pe_freq = "2"):
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
    
    if pe_freq == "2pi":
        scale = torch.pi
    else:
        scale = torch.pi/2

    if log_sampling == True:
        frequency_bands = scale * (freq ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        ))
    
    elif random_sampling == True:
        B = np.random.normal(loc = 0, scale = sigma, size = (int(num_encoding_functions/3), 3))
        frequency_bands = torch.pi * B

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

        else:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)

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
                    # Abstand zwischen Punkten: 2 * (1/64) = 1/32
                    # Sample zwischen 0 und 1; ziehe 0.5 ab und teile durch 32
            X[1:] = X[1:]+((np.random.random_sample(X[1:].shape)-0.5)/(self.resolution/2))
            X[0] = X[0]+((np.random.random_sample(X[0].shape)-0.5)/(self.z_resolution/2))
            # correct at boundaries
            X[X < -1] = -1
            X[X > 1] = 1 
        sample_idx = self.filter_mask[idx]
        
        sample = [X, sample_idx, y, self.weights]

        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
######################## METRICS #################################
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
    target_tmp = torch.flatten(target.float())

    tp = torch.sum(pred_tmp * target_tmp)
    fp = torch.sum(pred_tmp == (target_tmp + 1.))
    fn = torch.sum((pred_tmp + 1.) == target_tmp)
    union = tp+fp+fn
    iou = torch.div(tp, union)
    
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
    intersection = (input * target).sum()
    
    return (2. * intersection + smooth) / (input.sum() + target.sum() + smooth) 