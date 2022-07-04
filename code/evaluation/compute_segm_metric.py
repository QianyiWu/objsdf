"""Evaluation Metrics for Semantic Segmentation"""
from genericpath import exists
from re import I
import sys
from telnetlib import SE
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

__all__ = ['SegmentationMetric', 'batch_pix_accuracy', 'batch_intersection_union',
           'pixelAccuracy', 'intersectionAndUnion', 'hist_info', 'compute_score']


class SegmentationMetric(object):
    """
    Computes pixAcc and mIoU metric scores
    """

    def __init__(self, nclass):
        super(SegmentationMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)
        
    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    try:
        pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    except:
        print("predict size: {}, target size: {}, ".format(predict.size(), target.size()))
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled
    

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1  # [N,H,W] 
    target = target.float() + 1            # [N,H,W] 
    # print(predict.shape)
    # print(target.shape)

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def pixelAccuracy(imPred, imLab):
    """
    This function takes the prediction and label of a single image, returns pixel-wise accuracy
    To compute over many images do:
    for i = range(Nimages):
         (pixel_accuracy[i], pixel_correct[i], pixel_labeled[i]) = \
            pixelAccuracy(imPred[i], imLab[i])
    mean_pixel_accuracy = 1.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return (pixel_accuracy, pixel_correct, pixel_labeled)


def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)


def hist_info(pred, label, num_cls):
    assert pred.shape == label.shape
    k = (label >= 0) & (label < num_cls)
    labeled = np.sum(k)
    correct = np.sum((pred[k] == label[k]))

    return np.bincount(num_cls * label[k].astype(int) + pred[k], minlength=num_cls ** 2).reshape(num_cls,
                                                                                                 num_cls), labeled, correct


def compute_score(hist, correct, labeled):
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mean_IU = np.nanmean(iu)
    mean_IU_no_back = np.nanmean(iu[1:])
    freq = hist.sum(1) / hist.sum()
    freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
    mean_pixel_acc = correct / labeled

    return iu, mean_IU, mean_IU_no_back, mean_pixel_acc




def compute_sem_metric(**kwargs):
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

    # if eval_rendering:
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    collate_fn=all_dataset.collate_fn
                                                    )
    total_pixels = all_dataset.total_pixels
    img_res = all_dataset.img_res
    split_n_pixels = conf.get_int('train.split_n_pixels', 10000)

    old_checkpnts_dir = os.path.join(expdir, timestamp, 'checkpoints')
    print("Load checkpoint from {}".format(old_checkpnts_dir))

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']
    print("epoch: ", epoch)

    # calculate the model num
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("[INFO]: TOTAL NUMBER OF PARAMETER: {}".format(pytorch_total_params))

    model.eval()
    num_sem = conf.get_int('model.implicit_network.d_out')
    print("Number of segmentation class: ", num_sem)
    metric = SegmentationMetric(num_sem)
    total_mIOU = []
    total_pix_acc = []
    total_psnr = []
    # plot_dir = "vis_temp"
    # os.makedirs(plot_dir, exist_ok=True)
    # def get_plot_data(model_outputs, pose, rgb_gt, seg_gt):
    #     batch_size, num_samples, _ = rgb_gt.shape

    #     rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
    #     normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
    #     normal_map = (normal_map + 1.) / 2.
    #     semantic_map = model_outputs['semantic_map'].argmax(dim=-1).reshape(batch_size, num_samples, 1)

    #     plot_data = {
    #         'rgb_gt': rgb_gt,
    #         'pose': pose,
    #         'rgb_eval': rgb_eval,
    #         'normal_map': normal_map,
    #         'semantic_map': semantic_map,
    #         'seg_gt': seg_gt
    #     }

    #     return plot_data

    # with torch.no_grad():
    for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
        metric.reset()
        model_input["intrinsics"] = model_input["intrinsics"].cuda()
        model_input["uv"] = model_input["uv"].cuda()
        model_input['pose'] = model_input['pose'].cuda()

        split = utils.split_input(model_input, total_pixels, n_pixels=split_n_pixels)
        res = []
        for s in tqdm(split):
            torch.cuda.empty_cache()
            out = model(s)
            res.append({
                'rgb_values': out['rgb_values'].detach(),
                'normal_map': out['normal_map'].detach(),
                'semantic_map': out['semantic_values'].detach()
            })
        batch_size = ground_truth['rgb'].shape[0]
        model_output = utils.merge_output(res, total_pixels, batch_size)
        # plot_data = get_plot_data(model_output, model_input['pose'], ground_truth['rgb'], ground_truth['segs'])

        # plt.plot(model.implicit_network,
        #         indices,
        #         plot_data,
        #         plot_dir,
        #         epoch,
        #         img_res,
        #         **conf.get_config('plot')
        #         )
        sem_eval = model_output['semantic_map']
        # print(sem_eval.shape)
        sem_eval = sem_eval.reshape(img_res[0], img_res[1], -1).permute(2, 0, 1).unsqueeze(0)
        gt_sem = ground_truth['segs'].reshape(1, img_res[0], img_res[1]).cuda()
        # print(sem_eval)
        # print(gt_sem)
        # inter, union = batch_intersection_union(sem_eval, gt_sem, sem_eval.shape[-1])
        # IoU = 1.0 * inter / (2.220446049250313e-16 + union)
        # print(IoU.mean().item())
        metric.update(sem_eval, gt_sem)
        pix_acc, mIOU = metric.get()
        print(pix_acc, mIOU)
        total_mIOU.append(mIOU)
        total_pix_acc.append(pix_acc)
        psnr = rend_util.get_psnr(model_output['rgb_values'], ground_truth['rgb'].cuda().reshape(-1, 3))
        print(psnr)
        total_psnr.append(psnr)
        # print(ground_truth['segs'].shape)
        # assert False

    # total_mIOU = 
    # total_psnr = total_psnr/len(eval_dataloader)
    print('[INFO]: total mIOU: {}, pix acc: {}, total PSNR: {}'.format(sum(total_mIOU)/len(total_mIOU), sum(total_pix_acc)/len(total_pix_acc), sum(total_psnr)/len(total_psnr)))
    with open(os.path.join(old_checkpnts_dir, 'sem_metric_miou.txt'), 'w') as f:
        f.write('Miou\n')
        for i in total_mIOU:
            f.write(str(i)+'\n')
    with open(os.path.join(old_checkpnts_dir, 'sem_metric_pixacc.txt'), 'w') as f:
        f.write('pix_acc\n')
        for i in total_pix_acc:
            f.write(str(i)+'\n')
    with open(os.path.join(old_checkpnts_dir, 'sem_metric_psnr.txt'), 'w') as f:
        f.write('psnr\n')
        # f.write(total_psnr)
        for i in total_psnr:
            f.write(str(i)+'\n')
    
        






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

    opt = parser.parse_args()

    

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if (not gpu == 'ignore'):
        os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)

    compute_sem_metric(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name=opt.evals_folder,
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id=opt.scan_id,
             resolution=opt.resolution,
             eval_rendering=opt.eval_rendering,
             )