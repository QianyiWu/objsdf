import os
import trimesh 
import open3d as o3d
import sys
sys.path.append('../code')
import argparse
import GPUtil
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

import utils.general as utils
import utils.plots as plt
from utils import rend_util


def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst

def color_mesh(**kwargs):
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_rendering = kwargs['eval_rendering']

    expname = conf.get_string('train.expname') + kwargs['expname']
    scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else conf.get_int('dataset.scan_id', default=-1)
    if scan_id != -1:
        expname = expname + '_{0}'.format(scan_id)
    else:
        scan_id = conf.get_string('dataset.object', default='')

    if kwargs['timestamp'] == 'latest':
        if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname)):
            timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], expname))
            if (len(timestamps)) == 0:
                print('WRONG EXP FOLDER')
                exit()
            # self.timestamp = sorted(timestamps)[-1]
            timestamp = None
            for t in sorted(timestamps):
                if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], expname, t, 'checkpoints',
                                               'ModelParameters', str(kwargs['checkpoint']) + ".pth")):
                    timestamp = t
            if timestamp is None:
                print('NO GOOD TIMSTAMP')
                exit()
        else:
            print('WRONG EXP FOLDER')
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    all_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)
    eval_dataset = torch.utils.data.Subset(all_dataset, all_dataset.i_split[1])

    conf_model = conf.get_config('model')
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)
    if torch.cuda.is_available():
        model.cuda()

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    print("Load checkpoint from {}".format(old_checkpnts_dir))

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']
    print("epoch: ", epoch)

    # finish the model loading

    # load mesh
    trimesh_object = trimesh.load(kwargs['mesh_file'], process=False)
    o3d_mesh = trimesh_to_open3d(trimesh_object)
    vertices =  np.array(o3d_mesh.vertices).reshape(-1, 3).astype(np.float32)
    point_flat = torch.FloatTensor(vertices).cuda()
    ray_d = -torch.FloatTensor(np.asarray(o3d_mesh.vertex_normals)).cuda()
    ray_d = ray_d / torch.norm(ray_d, dim=-1, keepdim=True).float()
    print(ray_d.shape) # should be (-1, 3)

    _, feature_vectors, gradients, _, _ = model.implicit_network.get_outputs(point_flat)
    rgb_flat = model.rendering_network(point_flat, gradients, ray_d, feature_vectors)
    print(rgb_flat.shape)

    rgb_flat = (255*rgb_flat.detach().cpu()).numpy().astype(np.uint8)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(rgb_flat/255.0)
    o3d.io.write_triangle_mesh('color_mesh.ply', o3d_mesh)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--evals_folder', type=str, default='evals', help='The evaluation folder name.')
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiemnt timestamp to test.')
    parser.add_argument('--checkpoint', default='latest',type=str,help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--eval_rendering', default=False, action="store_true", help='If set, evaluate rendering quality.')
    parser.add_argument('--num_sem', default=6, help='Number of class')
    parser.add_argument('--mesh_file', type=str)

    opt = parser.parse_args()

    

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    color_mesh(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name=opt.evals_folder,
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             eval_rendering=opt.eval_rendering,
             mesh_file = opt.mesh_file
             )