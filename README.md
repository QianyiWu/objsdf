# Object-Compositional Neural Implicit Surfaces

### [Project Page](https://qianyiwu.github.io/objectsdf/) | [Paper](http://arxiv.org/abs/2207.09686)

![teaser](./media/teaser.gif)

This repository contains the official implementation of the ECCV2022 paper:  
**Object-Compositional Neural Implicit Surfaces**.  
[Qianyi Wu](https://qianyiwu.github.io/), [Xian Liu](https://alvinliu0.github.io/), [Yuedong Chen](https://donydchen.github.io/), [Kejie Li](https://likojack.github.io/kejieli/#/home), [Chuanxia Zheng](https://www.chuanxiaz.com/), [Jianfei Cai](https://jianfei-cai.github.io/), [Jianmin Zheng](https://personal.ntu.edu.sg/asjmzheng/).  
The paper introduces **ObjectSDF**: a volume rendering framework for object-compositional implicit neural surfaces, allowing learning high fidelity geometry of each object from a sparse set of input images and the corresponding semantic segmentation maps.

## Setup

#### Installation Requirements

The code is compatible with Python 3.9 and Pytorch 1.10.1. In addition, the following packages are required:
numpy, pyhocon, plotly, scikit-image, trimesh, imageio, opencv, torchvision.

You can create an anaconda environment called `objsdf` with the required dependencies by running:

```
conda env create -f environment.yml
conda activate objsdf
```

#### Data

We provide the installation guidance to use our code in the Toydesk dataset. First, you need to download [Toydesk](https://www.dropbox.com/s/bdqiv7pc13p6ugp/toydesk_data_full.zip?dl=0) dataset and put it in the './data' folder. Then

```
cd data
bash process_toydesk.sh
```

We require the RGB images with the corresponding semantic segmentation map for model training. After running the above script, you will get the corresponding file for two desk scenes.

## Usage

For example, if you would like to train ObjectSDF on Toydesk 2, please run:

```
cd ./code
python training/exp_runner.py --conf confs/toydesk2.conf --train_type objsdf
```

## Citation
If you use this project for your research, please cite our paper.

```bibtex
@inproceedings{wu2022object,
  title={Object-compositional neural implicit surfaces},
  author={Wu, Qianyi and Liu, Xian and Chen, Yuedong and Li, Kejie and Zheng, Chuanxia and Cai, Jianfei and Zheng, Jianmin},
  booktitle={European Conference on Computer Vision},
  pages={197--213},
  year={2022},
  organization={Springer}
}
```

## Related Links
If you are interested in **NeRF / neural implicit representations + semantic map**, we would also like to recommend you to check out our other related works:

* Neural implicit generative model, [Sem2NeRF](https://donydchen.github.io/sem2nerf/).

* Digital speaking portrait animation, [SSPNeRF](https://alvinliu0.github.io/projects/SSP-NeRF).

## Acknowledgments

Our implementation was mainly inspired by [VolSDF](https://github.com/lioryariv/volsdf).
