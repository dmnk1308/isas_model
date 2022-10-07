import numpy as np
from helpers import *
import skimage
import logging
import plyfile
#import open3d
import time
from tqdm import tqdm


# Code adapted from:  
# https://github.com/vsitzmann/siren/blob/master/sdf_meshing.py

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    from_mask = False
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.cpu().numpy()
    numpy_3d_sdf_tensor = np.float32(numpy_3d_sdf_tensor)
    
    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    if from_mask == True:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=voxel_size
        )
    else:
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing= voxel_size
        )


    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 1] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 0] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])
    
    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.text = True
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )

    )

def get_ply(model = None, mask = None, from_mask = False, ply_filename = "tmp", 
            resolution = 128, z_resolution = None, device = None, max_batch=64 ** 3, lung = None, features = None, slicewise = False, slices = None):
    offset=None
    scale=None
    ply_filename = "/home/dbecker/masterlung/visualization/ply_data/"+ply_filename+".ply"    
    if from_mask == True:
        print("Writing mask .ply file")
        mask = np.moveaxis(mask, 0, -1)
        mask = torch.from_numpy(mask)

        if slicewise == True:

            f_tmp = mask > 0
            f_tmp = torch.where(f_tmp)
            f_tmp = torch.stack(f_tmp,0)
            mask[f_tmp[0]+1,f_tmp[1]-1,f_tmp[2]] = 1
            mask[f_tmp[0]+1,f_tmp[1]+1,f_tmp[2]] = 1
            mask[f_tmp[0]-1,f_tmp[1]+1,f_tmp[2]] = 1
            mask[f_tmp[0]-1,f_tmp[1]-1,f_tmp[2]] = 1
            s = list(range(mask.shape[-1]))
            [s.remove(i) for i in slices]
            mask[:,:,s] = -1

        convert_sdf_samples_to_ply(mask, 
                                    voxel_grid_origin = [-1, -1, -1],
                                    voxel_size = [2.0 / (mask.shape[0]-1), 2.0 / (mask.shape[1]-1), 2.0 / (mask.shape[2]-1)],
                                    ply_filename_out = ply_filename, 
                                    from_mask = True)
        print("Done.")
        return None

    decoder = model
    decoder.to(device)
    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (resolution - 1)

    # transform first 3 columns
    # to be the x, y, z index
    v = torch.linspace(-1,1,resolution)
    if z_resolution == None:
        z_resolution = resolution
        vz = torch.linspace(-1,1,z_resolution)
    else:
        vz = torch.linspace(-1,1,z_resolution)

    samples = torch.zeros(resolution * resolution * z_resolution, 4).to(device)

    x,y,z = torch.meshgrid([v,v,vz], indexing = "xy")
    samples[:, 0] = torch.ravel(x).to(device)
    samples[:, 1] = torch.ravel(y).to(device)
    samples[:, 2] = torch.ravel(z).to(device)
    samples.requires_grad = False

    num_samples = resolution * resolution * z_resolution

    head = 0
    print("Writing model .ply file")
    while head < num_samples:
        idx = np.arange(start = head, stop = min(head + max_batch, num_samples))
        sample_subset = samples[idx,:3]
        
        if lung != None:
            sample_subset = (sample_subset.to(device), torch.full((len(idx),1), lung))# torch.tile(lung,(len(idx),1)).to(device))
        if torch.is_tensor(features):
            sample_subset = (sample_subset.to(device), features[idx].float().to(device))

        samples[idx, 3] = decoder(sample_subset).squeeze().detach()

        head += max_batch
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(resolution, resolution, z_resolution)

    voxel_size = [2.0 / (resolution-1), 2.0 / (resolution-1), 2.0 / (z_resolution-1)]

    convert_sdf_samples_to_ply(
        sdf_values.data,
        voxel_origin,
        voxel_size,
        ply_filename,
        offset,
        scale,
    )
    print("Done.")