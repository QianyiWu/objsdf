from re import T
import sys

sys.path.append('../code')
import argparse
import GPUtil

from training.volsdf_train import VolSDFTrainRunner
from training.objsdf_train import ObjSDFTrainRunner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument("--exps_folder", type=str, default="exps")
    parser.add_argument('--gpu', type=str, default='auto', help='GPU to use [default: GPU auto]')
    parser.add_argument('--is_continue', default=False, action="store_true",
                        help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str,
                        help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str,
                        help='The checkpoint epoch of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--cancel_vis', default=False, action="store_true",
                        help='If set, cancel visualization in intermediate epochs.')
    parser.add_argument('--finetune_folder', type=str, default=None, help='The path to finetune model')
    parser.add_argument('--train_type', default=str,
                        help='Use which type of trainer.')
    parser.add_argument('--finetune_file', type=str, default=None, help='the file contain the layers which gonna be finetuned.')
    parser.add_argument('--vis_seprate', default=False, action="store_true",
                        help='run rendering for each semantic label, only use if sdf is for each object')

    opt = parser.parse_args()

    if opt.gpu == "auto":
        deviceIDs = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False,
                                        excludeID=[], excludeUUID=[])
        gpu = deviceIDs[0]
    else:
        gpu = opt.gpu

    if opt.train_type=='origin':
        print("Using Original Trainer.")
        trainrunner = VolSDFTrainRunner(conf=opt.conf,
                                        batch_size=opt.batch_size,
                                        nepochs=opt.nepoch,
                                        expname=opt.expname,
                                        gpu_index=gpu,
                                        exps_folder_name=opt.exps_folder,
                                        is_continue=opt.is_continue,
                                        timestamp=opt.timestamp,
                                        checkpoint=opt.checkpoint,
                                        scan_id=opt.scan_id,
                                        do_vis=not opt.cancel_vis
                                        )
    elif opt.train_type=='objsdf':
        print("Using ObjectSDF Trainer.")
        trainrunner = ObjSDFTrainRunner(conf=opt.conf,
                                    batch_size=opt.batch_size,
                                    nepochs=opt.nepoch,
                                    expname=opt.expname,
                                    gpu_index=gpu,
                                    exps_folder_name=opt.exps_folder,
                                    is_continue=opt.is_continue,
                                    timestamp=opt.timestamp,
                                    checkpoint=opt.checkpoint,
                                    do_vis=not opt.cancel_vis,
                                    finetune_folder = opt.finetune_folder,
                                    finetune_file = opt.finetune_file
                                    )
    else:
        raise ValueError('opt.train_type {} is not implemented'.format(opt.train_type))

    trainrunner.run()