from configparser import Interpolation
from curses.ascii import NUL
import os
import torch
import numpy as np
from tqdm import tqdm
import json
import random
from PIL import Image
from torchvision.transforms import functional as F
import imageio
import cv2

class ScanNetDataset(torch.utils.data.Dataset):
    """
    Dataset class for ScanNet dataset
    """
    def __init__(self,
                 data_dir,
                 img_res,
                 half_res = True,
                 instance = True):
        
        self.train_list = np.loadtxt(os.path.join(data_dir, 'train.txt')).astype(int).tolist()
        self.test_list = np.loadtxt(os.path.join(data_dir, 'test.txt')).astype(int).tolist()

        self.img_res = img_res
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.sampling_idx = None
        
        all_imgs = []
        all_sems = []
        all_poses = []

        self.instance = instance

        img_dir = os.path.join(data_dir, 'color')
        if instance:
            sem_dir = os.path.join(data_dir, 'instance-filt')
        else:
            sem_dir = os.path.join(data_dir, 'label-filt')
        pose_dir = os.path.join(data_dir, 'pose')

        self.label_mapping = None
        self.instance_mapping_dict= {}
        if instance:
            # with open(os.path.join(data_dir, 'label_mapping_instance.txt'), 'r') as f:
            #     content = f.readlines()
            #     self.label_mapping = [int(a) for a in content[0].split(',')]

            # using the remapped instance label for training
            with open(os.path.join(data_dir, 'instance_mapping.txt'), 'r') as f:
                for l in f:
                    (k, v_sem, v_ins) = l.split(',')
                    self.instance_mapping_dict[int(k)] = int(v_ins)
            self.label_mapping = [] # get the sorted label mapping list
            for k, v in self.instance_mapping_dict.items():
                if v not in self.label_mapping: # not a duplicate instance
                    self.label_mapping.append(v)
            print('Instance Label Mapping: ', self.label_mapping)
        else:
            with open(os.path.join(data_dir, 'label_mapping.txt'), 'r') as f:
                content = f.readlines()
                self.label_mapping = [int(a) for a in content[0].split(',')]
        # load center file 
        self.center = None
        if os.path.exists(os.path.join(data_dir, 'center.txt')):
            self.center = np.loadtxt(os.path.join(data_dir, 'center.txt')).reshape(4,1)
        self.center_mat = np.zeros([4, 4])
        self.scale_mat = np.eye(4)
        # print(self.center.shape)
        if self.center is not None:
            self.center_mat[:3, 3:] = self.center[:3]
            self.scale_mat[0, 0] = 1.0/self.center[-1]
            self.scale_mat[1, 1] = 1.0/self.center[-1]
            self.scale_mat[2, 2] = 1.0/self.center[-1]
        # print(self.center_mat)
        train_imgs, train_segs, train_poses = self.load_meta_data(os.path.join(data_dir, 'train.txt'), img_dir, sem_dir, pose_dir)
        test_imgs, test_segs, test_poses = self.load_meta_data(os.path.join(data_dir, 'test.txt'), img_dir, sem_dir, pose_dir)
        
        train_imgs = (np.array(train_imgs) / 255.).astype(np.float32)
        train_segs = np.array(train_segs).astype(np.float32)
        train_poses = np.array(train_poses).astype(np.float32)
        test_imgs = (np.array(test_imgs) / 255.).astype(np.float32)
        test_segs = np.array(test_segs).astype(np.float32)
        test_poses = np.array(test_poses).astype(np.float32)
        all_imgs.append(train_imgs)
        all_imgs.append(test_imgs)
        all_sems.append(train_segs)
        all_sems.append(test_segs)
        all_poses.append(train_poses)
        all_poses.append(test_poses)

        self.i_split = [np.array(i) for i in [range(len(train_imgs)), range(len(train_imgs), len(train_imgs)+len(test_imgs))]]
        self.imgs = np.concatenate(all_imgs, 0)
        self.segs = np.concatenate(all_sems, 0)
        self.poses = np.concatenate(all_poses, 0)
        self.n_imgs = len(self.i_split)

        self.intrinsics_all = []
        color_intrisic_path = os.path.join(data_dir, 'intrinsic', 'intrinsic_color.txt')
        with open(color_intrisic_path, 'r') as f:
            lines = [x.strip() for x in f.readlines() if x.strip()]
        focal = float(lines[0].split(' ')[0])
        # Original resolution
        # for pose in self.poses:
        #     self.intrinsics_all.append(torch.from_numpy(np.array([
        #                                     [1163.445068, 0.000000, 653.626038, 0],
        #                                     [.000000, 1164.793945, 481.600037, 0],
        #                                     [0, 0, 1, 0],
        #                                     [0, 0, 0, 1]
        #                                     ])).float())

        ins = []
        for i in range(4):
            ins.append([float(a) for a in lines[i].split(' ')])
        ins = np.array(ins)
        if half_res:
            # Half resolution for limit RAM
            # change the resolution from original [968, 1296] to [480, 640]
            # ins[0] = ins[0]/2
            # ins[1] = ins[1]/2
            ins[0] = ins[0] / 1296 * img_res[1]
            ins[1] = ins[1] / 968 * img_res[0]
        for pose in self.poses:
            self.intrinsics_all.append(torch.from_numpy(ins).float())


    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": torch.from_numpy(self.poses[idx]).float()
        }

        ground_truth = {
            "rgb": torch.from_numpy(self.imgs[idx]).float(),
            "segs": torch.from_numpy(self.segs[idx]).float()
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = torch.from_numpy(self.imgs[idx][self.sampling_idx, :])
            ground_truth["segs"] = torch.from_numpy(self.segs[idx][self.sampling_idx, :])
            sample["uv"] = uv[self.sampling_idx, :]
        return idx, sample, ground_truth
    
    def __len__(self):
        return self.n_images


    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def load_meta_data(self, split_path, img_dir, seg_dir, pose_dir):
        fix_rot = np.array([1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1]).reshape(4, 4)

        with open(split_path, 'r') as f:
            contents = f.readlines()
            img_ids = [x.strip() for x in contents if x.strip()]
        split_imgs = []
        split_sems = []
        split_poses = []
        for cur_id in img_ids:
            split_imgs.append((cv2.resize(imageio.imread(os.path.join(img_dir, "%s.jpg" % str(cur_id))), (self.img_res[1], self.img_res[0]), interpolation=cv2.INTER_AREA))\
                .transpose(2, 0, 1).reshape(3, -1).transpose(1, 0))

            ori_sems = cv2.resize(imageio.imread(os.path.join(seg_dir, "%s.png" % str(cur_id))), (self.img_res[1], self.img_res[0]), interpolation=cv2.INTER_NEAREST).\
                reshape(1, -1).transpose(1, 0)
            
            ins_list = np.unique(ori_sems)
            cur_sems = np.copy(ori_sems)
            if self.label_mapping is not None:
                for i in ins_list if self.instance else self.label_mapping:
                    # cur_sems[cur_sems == i] = self.label_mapping.index(i)
                    cur_sems[ori_sems == i] = self.label_mapping.index(self.instance_mapping_dict[i]) if self.instance else self.label_mapping(i)
            split_sems.append(cur_sems)

            pose_path = os.path.join(pose_dir, "%s.txt" % str(cur_id))
            cur_pose = []
            with open(pose_path, 'r') as f:
                contents = f.readlines()
                lines = [x.strip() for x in contents if x.strip()]
            for line in lines:
                cur_pose.append([float(x) for x in line.split(' ')])
            # centerize pose by self.center
            pose_matrix = np.array(cur_pose, dtype=float) - self.center_mat
            pose_matrix = self.scale_mat @ pose_matrix
            split_poses.append(pose_matrix @ fix_rot)
        
        return split_imgs, split_sems, split_poses
