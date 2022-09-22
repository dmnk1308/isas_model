import numpy as np
from helpers import *
import skimage
import logging
import plyfile
#import open3d
import time
from tqdm import tqdm



# From: https://github.com/vsitzmann/siren/blob/master/sdf_meshing.py
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
    
def get_mesh(ply_filename, ply_filename_mask = None, save_gif = True, gif_filename = None, frames = 100, 
             duration = 5, loop = 0, numbering = False, rotate = True,
             output_window = True, both = False):
    """Function to output open3d window or save gif of rotating mesh

    Args:
        ply_filename (_type_): _description_
        save_gif (bool, optional): _description_. Defaults to True.
        gif_filename (_type_, optional): _description_. Defaults to None.
    """
    m = 1
    ply_filename = "visualization/ply_data/"+ply_filename+".ply"
    try:
        ply_filename_mask = "visualization/ply_data/"+ply_filename_mask+".ply"
    except:
        None
    
    # load mesh data
    mesh = open3d.io.read_triangle_mesh(ply_filename) 
    mesh.compute_vertex_normals()
    # rotate to a view from the front of the lung
    # 90 degree rotation = pi/2
    R = mesh.get_rotation_matrix_from_xyz((1*np.pi / 2, 0, 3*np.pi / 2))
    mesh = mesh.rotate(R, center=(0,0,0))
    #mesh = mesh.paint_uniform_color([96/255, 96/255, 96/255])
    c = mesh.get_center()

    
    if both == True:
        # make window double width
        m = 2
        mesh = mesh.translate((-1,0,0))
        c = mesh.get_center()

        # load mesh data for mask
        mesh_mask = open3d.io.read_triangle_mesh(ply_filename_mask) 
        mesh_mask.compute_vertex_normals()
        # rotate to a view from the front of the lung                        
        R = mesh_mask.get_rotation_matrix_from_xyz((1*np.pi / 2, 0, 3*np.pi / 2))
        mesh_mask = mesh_mask.rotate(R, center=(0,0,0))
        mesh_mask = mesh_mask.translate((1,0,0))
        c_mask = mesh_mask.get_center()
        
    if save_gif == True:
        imgs = []
        # make screenshots after each rotation move
        for i in tqdm(range(frames)):
            
            #R = mesh.get_rotation_matrix_from_axis_angle((0,0.08,0))
            R = mesh.get_rotation_matrix_from_xyz((0,0.1*np.pi / 2,0))           
            mesh = mesh.rotate(R, center= c)
            mesh.translate(c, relative = False)
            if both == True:
                R = mesh_mask.get_rotation_matrix_from_axis_angle((0,0.08,0))
                mesh_mask = mesh_mask.rotate(R, center=c_mask)
                mesh_mask.translate(c_mask, relative = False)

            vis = open3d.visualization.Visualizer()
            vis.create_window(visible=False, width = m * 512, height = 512)
            vis.add_geometry(mesh)
            vis.update_geometry(mesh)

            if both == True:
                vis.add_geometry(mesh_mask)
                vis.update_geometry(mesh_mask)
            
            vis.poll_events()
            vis.update_renderer()
            #depth = vis.capture_depth_float_buffer(do_render=False)
            imgs.append(np.array(vis.capture_screen_float_buffer(do_render=False)))
            vis.destroy_window()
        
        make_gif(imgs, gif_filename, convert = False, mesh = True, duration = duration, loop = loop, numbering = numbering)

    if output_window == True:
        if rotate == True:
            def rotate_view(vis):
                ctr = vis.get_view_control()
                ctr.rotate(5.0, 0.0) # rotates 10 degrees every frame
                
            if both == True:
                meshes = [mesh, mesh_mask]
                open3d.visualization.draw_geometries_with_animation_callback(meshes, rotate_view)
            else:
                open3d.visualization.draw_geometries_with_animation_callback([mesh], rotate_view)
        else:
            vis = open3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)
            vis.add_geometry(mesh_mask) if both == True else None
            vis.run()
            vis.destroy_window()
            
            
def visualize_model(model = None, mask = None, from_mask = False, both = False, resolution = 128, device = None, max_batch=64 ** 3, save_gif = True, 
                    gif_filename = "lung.gif", frames = 100, rewrite_ply = True, rotate = True, ply_filename = "tmp",
                    duration = 5, loop = 0, numbering = False, output_window = True, lung = None):

    gif_filename = "visualization/mesh_gifs/"+gif_filename+".gif"
    
    if from_mask == True:
        ply_filename_mask = "tmp_mask"
        
    # make .ply
    # write .ply for mask
    if np.all((from_mask == True, both == False)):
        if rewrite_ply == True:
            get_ply(mask = mask, ply_filename = ply_filename_mask, from_mask = True, resolution = resolution, device = device, max_batch = 64**3, lung = lung)
    
    # write .ply for mask & model
    elif both == True:    
        if rewrite_ply == True:
            get_ply(mask = mask, ply_filename = ply_filename_mask, from_mask = True, resolution = resolution, device = device, max_batch = 64**3, lung = lung)

        if rewrite_ply == True:    
            get_ply(model = model, ply_filename = ply_filename, resolution = resolution, device = device, max_batch = 64**3, lung = lung)
            
    # write .ply for model
    else:
        if rewrite_ply == True:    
            get_ply(model = model, ply_filename = ply_filename, resolution = resolution, device = device, max_batch = 64**3, lung = lung)

    # visualize
    if both == False:
        get_mesh(ply_filename, save_gif = save_gif, gif_filename = gif_filename, frames = frames, 
            duration = duration, loop = loop, numbering = numbering, rotate = rotate, 
            output_window = output_window)
    else:
        get_mesh(ply_filename = ply_filename, ply_filename_mask = ply_filename_mask, save_gif = save_gif, gif_filename = gif_filename, frames = frames, 
            duration = duration, loop = loop, numbering = numbering, rotate = rotate, 
            output_window = output_window, both = both)