import os
import torch
import numpy as np

# import utils.general as utils
from tqdm import tqdm
import json
import random
from PIL import Image
from torchvision.transforms import functional as F
import imageio

class ToydeskDataset(torch.utils.data.Dataset):
    """
    Dataset class for toydesk dataset
    """
    def __init__(self,
                 data_dir,
                 img_res):
        
        self.meta = {}
        with open(os.path.join(data_dir, "transforms_full.json"), 'r') as f:
            self.meta = json.load(f)
        self.train_list = np.loadtxt(os.path.join(data_dir, 'train.txt')).astype(int).tolist()
        self.test_list = np.loadtxt(os.path.join(data_dir, 'test.txt')).astype(int).tolist()

        self.img_res = img_res
        self.total_pixels = self.img_res[0] * self.img_res[1]
        self.sampling_idx = None

        train_imgs = []
        train_segs = []
        train_poses = []
        test_imgs = []
        test_segs = []
        test_poses = []
        all_imgs = []
        all_sems = []
        all_poses = []
        # fix_rot = np.array([1, 0, 0, 0,
        #                     0, -1, 0, 0,
        #                     0, 0, -1, 0,
        #                     0, 0, 0, 1]).reshape(4, 4)
        fix_rot = np.array([1, 0, 0, 0,
                            0, 1, 0, 0,
                            0, 0, 1, 0,
                            0, 0, 0, 1]).reshape(4, 4)
        
        self.label_mapping = None
        with open(os.path.join(data_dir, 'label_mapping.txt'), 'r') as f:
            content = f.readlines()
            self.label_mapping = [int(a) for a in content[0].split(',')]


        self.center = None
        if os.path.exists(os.path.join(data_dir, 'center.txt')):
            self.center = np.loadtxt(os.path.join(data_dir, 'center.txt')).reshape(4, 1)
        self.center_mat = np.zeros([4, 4])
        self.scale_mat = np.eye(4)
        if self.center is not None:
            self.center_mat[:3, 3:] = self.center[:3]
            self.scale_mat[0, 0] = 1.0/self.center[-1]
            self.scale_mat[1, 1] = 1.0/self.center[-1]
            self.scale_mat[2, 2] = 1.0/self.center[-1]

        for frame in self.meta['frames'][::1]:
            if frame['idx'] in self.train_list:
                img_fname = os.path.join(data_dir, frame['file_path'])+'.png'
                seg_fname = os.path.join(data_dir, frame['file_path'])+'.instance.png'
                train_imgs.append(imageio.imread(img_fname).transpose(2, 0, 1).reshape(3, -1).transpose(1, 0)) # reshape to HW*3
                segs = imageio.imread(seg_fname).reshape(1, -1).transpose(1, 0)
                if self.label_mapping is not None:
                    for i in self.label_mapping:
                        segs[segs == i] = self.label_mapping.index(i)
                train_segs.append(segs) # reshape to HW*1
                pose_matrix = np.array(frame['transform_matrix'] + self.center_mat)
                pose_matrix = self.scale_mat @ pose_matrix
                train_poses.append( pose_matrix @ fix_rot)
            elif frame['idx'] in self.test_list:
                img_fname = os.path.join(data_dir, frame['file_path'])+'.png'
                seg_fname = os.path.join(data_dir, frame['file_path'])+'.instance.png'
                test_imgs.append(imageio.imread(img_fname).transpose(2, 0, 1).reshape(3, -1).transpose(1, 0))
                segs = imageio.imread(seg_fname).reshape(1, -1).transpose(1, 0)
                if self.label_mapping is not None:
                    for i in self.label_mapping:
                        segs[segs == i] = self.label_mapping.index(i)
                test_segs.append(segs)

                pose_matrix = np.array(frame['transform_matrix'] + self.center_mat)
                pose_matrix= self.scale_mat @ pose_matrix
                test_poses.append( pose_matrix @ fix_rot)
            else:
                continue
        
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
        # print(self.poses.max(axis=0))
        # print(self.poses.min(axis=0))
        self.n_imgs = len(self.i_split)

        self.intrinsics_all = []
        focal = 0.5 * img_res[1] / np.tan(0.5* float(self.meta['camera_angle_x']))
        for pose in self.poses:
            self.intrinsics_all.append(torch.from_numpy(np.array([
                                            [focal, 0, 0.5*self.img_res[1], 0],
                                            [0, focal, 0.5*self.img_res[0], 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]
                                            ])).float())

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

    def fix_sampling_pattern(self, sampling_idx): 
        self.sampling_idx = sampling_idx

if __name__ == '__main__':
    a = ToydeskDataset('/home/monster/Projects/Nerf_experiment/data/toydesk_data/processed/our_desk_2', [640, 480])
    i_split = a.i_split
    print(i_split)
    # train_split = torch.utils.data.sampler.SubsetRandomSampler(i_split[0])
    # test_split = torch.utils.data.sampler.SubsetRandomSampler(i_split[1])
    test_data = torch.utils.data.Subset(a, i_split[1])
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=2, shuffle=True, collate_fn=a.collate_fn)
    for epoch in range(3):
        if epoch == 0:
            a.change_sampling_idx(1024)
        else:
            a.change_sampling_idx(-1)
        for batch_idx, (idx, sample, gt) in enumerate(test_loader):
            print(epoch)
            print(idx, sample['uv'])