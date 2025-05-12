# GLFM-Multi-class-3DAD

> [**IEEE TASE**] [**Boosting Global-Local Feature Matching via Anomaly Synthesis for Multi-Class Point Cloud Anomaly Detection**](https://export.arxiv.org/abs/2409.13162).
>
> by [Yuqi Cheng](https://hustcyq.github.io/), [Yunkang Cao](https://caoyunkang.github.io/), Dongfang Wang, [Weiming Shen](https://scholar.google.com/citations?user=FuSHsx4AAAAJ&hl=en), Wenlong Li

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boosting-global-local-feature-matching-via/3d-anomaly-detection-and-segmentation-on)](https://paperswithcode.com/sota/3d-anomaly-detection-and-segmentation-on?p=boosting-global-local-feature-matching-via)

## Introduction 
Point cloud anomaly detection is essential for various industrial applications. The huge computation and storage costs caused by the increasing product classes limit the application of single-class unsupervised methods, necessitating the development of multi-class unsupervised methods. However, the feature similarity between normal and anomalous points from different class data leads to the feature confusion problem, which greatly hinders the performance of multi-class methods. Therefore, we introduce a multi-class point cloud anomaly detection method, named GLFM, leveraging global-local feature matching to progressively separate data that are prone to confusion across multiple classes. Specifically, GLFM is structured into three stages: Stage-I proposes an anomaly synthesis pipeline that stretches point clouds to create abundant anomaly data that are utilized to adapt the point cloud feature extractor for better feature representation. Stage-II establishes the global and local memory banks according to the global and local feature distributions of all the training data, weakening the impact of feature confusion on the establishment of the memory bank. Stage-III implements anomaly detection of test data leveraging its feature distance from global and local memory banks. Extensive experiments on the MVTec 3D-AD, Real3D-AD and actual industry parts dataset showcase our proposed GLFM’s superior point cloud anomaly detection performance.

## Overview of GLFM
<img src="./image/overview.png" width="800px">




## 🛠️ Getting Started

### Installation
To set up the GLFM environment, follow the methods below:

- Clone this repo:
```shell
git clone https://github.com/hustCYQ/GLFM.git && cd GLFM
```
- Construct the experimental environment, follow these steps:
```shell
conda create --name GLFM python=3.8
conda activate GLFM
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install tifffile open3d-cpu
pip install -r requirements.txt

pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt
pip install -e .
cd ..
```


### Dataset Preparation 
Welcome to directly download our **processed** datasets. All datasets need to be placed in your `DATA_ROOT`.

| Dataset | Google Drive | Baidu Drive | Note
|------------|------------------|------------------| ------------------|
| MVTec 3D-AD    | [Google Drive]() | [Baidu Drive]() | Remove Background |
| Real3D-AD    | [Google Drive]() | [Baidu Drive]() | Cut Training Data |


### Self-Supervised
```shell
python train.py
```



### Train & Test

```
python main.py --dataset mvtec --task Single-Class --k_class 1
python main.py --dataset real --task Single-Class --k_class 1
python main.py --dataset mvtec --task Multi-Class --k_class 10
python main.py --dataset real --task Multi-Class --k_class 3
```

### Checkpoints
We offer [pointmae_adapted.pth](https://pan.baidu.com/s/1O25jxtqo1yfQXhv_yjHwPg?pwd=gamw) for evaluation and [pointmae_pretrain.pth](https://pan.baidu.com/s/1We1JvqCXjgR9Bsc7idzucQ?pwd=fda7) for Self-Supervised.


## Main Results

Notes: We achieved good performance in Real3D when we converted Real3D to .tiff format. The performance below results from training on MVtec 3D and evaluating directly on raw points of Real3D, so it is lower than the report in the paper.
### Multi-Class Task

#### MVTec 3D-AD
| GLFM   |   Bagel |   Cable_Gland |   Carrot |   Cookie |   Dowel |   Foam |   Peach |   Potato |   Rope |   Tire |   Mean |
|:---------|--------:|--------------:|---------:|---------:|--------:|-------:|--------:|---------:|-------:|-------:|-------:|
| Image ROCAUC     |   0.958 |          0.78 |     0.99 |    0.992 |   0.953 |  0.837 |   0.931 |    0.983 |  0.986 |  0.994 |   0.94 |
| Pixel ROCAUC     |   0.989 |         0.974 |    0.998 |    0.935 |   0.958 |   0.94 |   0.998 |    0.999 |  0.996 |  0.997 |  0.978 |
| AU PRO     |   0.967 |         0.903 |    0.981 |    0.897 |   0.882 |  0.778 |    0.98 |    0.983 |  0.957 |  0.978 |  0.931 |


#### Real3D-AD
| GLFM   |   Airplane |   Candybar |   Car |   Chicken |   Diamond |   Duck |   Fish |   Gemstone |   Seahorse |   Shell |   Starfish |   Toffees |   Mean |
|:---------|-----------:|-----------:|------:|----------:|----------:|-------:|-------:|-----------:|-----------:|--------:|-----------:|----------:|-------:|
| Image ROCAUC     |      0.672 |      0.635 | 0.549 |     0.554 |     0.475 |  0.835 |  0.603 |      0.528 |       0.87 |   0.336 |      0.685 |     0.586 |  0.611 |
| Pixel ROCAUC     |      0.654 |      0.737 | 0.704 |     0.534 |     0.669 |  0.601 |  0.777 |      0.465 |      0.753 |   0.477 |       0.58 |     0.838 |  0.649 |


### Single-Class Task

#### MVTec 3D-AD
| GLFM   |   Bagel |   Cable_Gland |   Carrot |   Cookie |   Dowel |   Foam |   Peach |   Potato |   Rope |   Tire |   Mean |
|:---------|--------:|--------------:|---------:|---------:|--------:|-------:|--------:|---------:|-------:|-------:|-------:|
| Image ROCAUC     |   0.957 |         0.789 |    0.993 |    0.992 |   0.954 |   0.84 |   0.932 |    0.978 |  0.984 |  0.992 |  0.941 |
| Pixel ROCAUC     |   0.988 |         0.973 |    0.998 |    0.935 |   0.958 |   0.94 |   0.998 |    0.999 |  0.996 |  0.997 |  0.978 |
| AU PRO     |   0.965 |         0.897 |    0.981 |    0.897 |   0.882 |  0.777 |    0.98 |    0.983 |  0.957 |  0.978 |   0.93 |


#### Real3D-AD
| GLFM   |   Airplane |   Candybar |   Car |   Chicken |   Diamond |   Duck |   Fish |   Gemstone |   Seahorse |   Shell |   Starfish |   Toffees |   Mean |
|:---------|-----------:|-----------:|------:|----------:|----------:|-------:|-------:|-----------:|-----------:|--------:|-----------:|----------:|-------:|
| Image ROCAUC     |      0.474 |      0.722 | 0.661 |     0.607 |      0.43 |  0.934 |  0.627 |      0.622 |      0.939 |   0.664 |      0.723 |     0.744 |  0.679 |
| Pixel ROCAUC     |      0.617 |      0.826 | 0.762 |     0.519 |     0.727 |  0.651 |  0.905 |      0.452 |      0.809 |   0.474 |      0.533 |     0.937 |  0.684 |




If you find this repository useful for your research, please use the following.

```
@ARTICLE{GLFM,
  author={Cheng, Yuqi and Cao, Yunkang and Wang, Dongfang and Shen, Weiming and Li, Wenlong},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={Boosting Global-Local Feature Matching via Anomaly Synthesis for Multi-Class Point Cloud Anomaly Detection}, 
  year={2025},
  volume={22},
  number={},
  pages={12560-12571},
  keywords={Feature extraction;Point cloud compression;Anomaly detection;Data models;Training;Image reconstruction;Computational modeling;Automation;Self-supervised learning;Pipelines;Anomaly detection;point cloud;multi-class;global-local feature matching;anomaly synthesis},
  doi={10.1109/TASE.2025.3544462}}
```


## Acknowledgments
- This work was supported in part by the Ministry of Industry and Information Technology of the People’s Republic of China under Grant 2023ZY01089 and
 in part by the Fundamental Research Funds for the Central Universities of  China under Grant HUST: 2021GCRC058.

- Our code implementation is based on [3D-ADS](https://github.com/eliahuhorwitz/3D-ADS) and [M3DM](https://github.com/nomewang/M3DM), thanks for your good jobs.
