import os
import random
import imageio
import sys
import numpy as np
from random import sample

random.seed(10)

if __name__ == '__main__':
    label_set = set({})
    for f in os.listdir(os.path.join(sys.argv[1], 'instance-filt')):
        cur_sems = imageio.imread(os.path.join(sys.argv[1], 'instance-filt', f))
        # print(set(cur_sems.flatten()))
        label_set = set.union(label_set, set(cur_sems.flatten()))
        print(sorted(label_set))
    print(sorted(label_set))
    with open(os.path.join(sys.argv[1], 'label_mapping_instance.txt'), 'w') as f:
        f.write(str(sorted(label_set)).strip('[').strip(']'))
    f.close()
    
    # with open(os.path.join(sys.argv[1], 'label_mapping.txt'), 'r') as f:
    #     content = f.readlines()
    #     print([int(a) for a in content[0].split(',')])
    
    # print("split train/test")
    # name_list = []
    # for f in os.listdir(os.path.join(sys.argv[1], 'color')):
    #     name_list.append(f.split('.')[0])
    # test_list = sample(name_list, int(len(name_list)*0.1))
    # train_list = list(set(name_list).difference(set(test_list)))
    # print(train_list, test_list)
    # with open(os.path.join(sys.argv[1], 'train.txt'), 'w') as f:
    #     for inde in train_list:
    #         f.write(str(inde) + '\n')
    # f.close()
    # with open(os.path.join(sys.argv[1], 'test.txt'), 'w') as f:
    #     for inde in test_list:
    #         f.write(str(inde) + '\n')
    # f.close()

    # poses = []
    # for f in os.listdir(os.path.join(sys.argv[1], 'pose')):
    #     poses.append(np.loadtxt(os.path.join(sys.argv[1], 'pose', f)))
        
    # poses = np.array(poses)
    # print(np.min(poses, axis=0)[:3,3:])
    # print(np.max(poses, axis=0)[:3,3:])
