from dis import dis
import sys
from matplotlib.transforms import Transform
import torch
# Chamfer Distance Code borrowed from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch. Please follow their instructions to install the chamfer distance.
import chamfer3D.dist_chamfer_3D 
import trimesh
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise ValueError("Usage: \n \
            python computer_3d_metric.py path_to_gt_mesh path_to_generate_mesh center_file")
    # init chamfer distance 
    chamDiss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    
    # toydesk
    # box = trimesh.creation.box(extents=[0.7, 0.7, 0.7])
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])  # a predefined crop size for toydesk data, you can change it to your own crop size

    # print(box.facets_origin)
    # Load gt mesh
    gt_mesh_path = sys.argv[1]
    # print(gt_mesh_path)
    mesh = trimesh.load(gt_mesh_path)
    # rescale the gt_mesh to match the size
    scale_para = np.loadtxt(sys.argv[3]) # load center.txt
    mesh.vertices = (mesh.vertices + scale_para[:3]) / scale_para[-1] 
    # gt_mesh = mesh.convex_hull
    gt_mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)
    gt_mesh_new = mesh.slice_plane(box.facets_origin, -box.facets_normal)
    # components = gt_mesh_new.split(only_watertight=False)
    # areas = np.array([c.area for c in components], dtype=np.float32)
    # gt_mesh_new = components[areas.argmax()]
    gt_mesh_new.export('gt.ply') # visualize the gt mesh after crop

    # Load generate mesh
    gen_mesh_path = sys.argv[2]
    g_mesh = trimesh.load(gen_mesh_path)
    # scale for generation results like semantic-nerf, which didn't do the scale normalization during training.
    # g_mesh.vertices = (g_mesh.vertices - scale_para[:3]) / scale_para[-1]
    # g_vertices = torch.from_numpy(np.array(g_mesh.vertices)).unsqueeze(0).cuda()
    # Taking the biggest connected component if needed
    mesh_clean = g_mesh.slice_plane(box.facets_origin, -box.facets_normal)
    components = mesh_clean.split(only_watertight=False)
    areas = np.array([c.area for c in components], dtype=np.float32)
    mesh_clean = components[areas.argmax()]

    mesh_clean.export('test.ply')

    # dist1: from gt to generate
    dist1, dist2, idx1, idx2 = chamDiss(torch.from_numpy(gt_mesh_new.vertices).unsqueeze(0).float().cuda(),  \
        torch.from_numpy(mesh_clean.vertices).unsqueeze(0).float().cuda())

    print(dist1.mean())
    print(dist2.mean())
    print('Chamfer Distance: ', scale_para[-1]*0.5*(dist1.mean()+dist2.mean())) # scale to the original size



