# Object-Compositional Neural Implicit Surfaces

### [Project Page](https://qianyiwu.github.io/objectsdf/) | [Paper](https://qianyiwu.github.io/objectsdf/)

![teaser](./media/teaser.gif)

This repository contains an implementation for the paper:  
**Object-Compositional Neural Implicit Surfaces**.  
[Qianyi Wu](https://qianyiwu.github.io/), [Xian Liu](https://alvinliu0.github.io/), [Yuedong Chen](https://donydchen.github.io/), [Kejie Li](https://likojack.github.io/kejieli/#/home), [Chuanxia Zheng](https://www.chuanxiaz.com/), [Jianfei Cai](https://jianfei-cai.github.io/), [Jianmin Zheng](https://personal.ntu.edu.sg/asjmzheng/).  
The paper introduce **ObjectSDF**: a volume rendering framework for object-compositional implicit neural surfaces, allowing to learn high fidelity geometry of each object from a sparse set of input images and corresponding semantic segmentation map.

## Setup

#### Installation Requirmenets

The code is compatible with python 3.9 and pytorch 1.10.1. In addition, the following packages are required:
numpy, pyhocon, plotly, scikit-image, trimesh, imageio, opencv, torchvision.

You can create an anaconda environment called `objsdf` with the required dependencies by running:

```
conda env create -f environment.yml
conda activate objsdf
```

#### Data

We provide the installation guidence to use our code in Toydesk dataset. At first, you need to download [Toydesk](https://zjueducn-my.sharepoint.com/:u:/g/personal/ybbbbt_zju_edu_cn/EVgSQo48M6ZNmNqrggYU0qUBqSlAjOTMYn1YuRztdM2uTw?e=icdHdp) dataset and put it in the './data' folder. Then

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
