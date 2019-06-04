## 运行环境配置
添加 channel：
```shell
conda config --append channels conda-forge
```
首先创建虚拟环境：
```shell
conda create -n spine python=3.6.8 openmpi -y
```
安装 torch 和 torchvision：
```shell
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-linux_x86_64.whl
pip install scipy scikit-image scikit-learn matplotlib pandas requests tqdm SimpleITK gluoncv-torch
```
安装 apex：
```shell
git clone https://github.com/NVIDIA/apex.git
cd apex
CUDA_HOME=<path to cuda 10> pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```
其中 `CUDA_HOME` 为安装在 `/usr/local` 目录下的 cuda 的路径，版本号与 conda 安装的 cudatoolkit 要一致．本项目使用 cuda 10

安装 nccl 和 horovod．nccl 从 NVIDIA 官网下载 nccl，注意选择对应的版本号．无管理员权限的使用 tarball 解压即可．更多信息请参考 horovod 的官网．
```shell
tar xfv nccl_2.4.2-1+cuda10.0_x86_64.txz
HOROVOD_CUDA_HOME=<path to cuda 10> HOROVOD_NCCL_HOME='pwd/nccl_2.4.2-1+cuda10.0_x86_64' HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod
```
## 训练
使用 4 个 GPU 进行训练：
```shell
CUDA_VISIBLE_DEVICES="0,1,2,3" horovodrun -np 4 -H localhost:4 python train_distributed.py --network UNet --backbone resnet50 --save_model_path checkpoint
```
## 在验证集上验证并计算结果
```shell
python eval.py --network UNet --backbone resnet50 --model  --result_dir result
```
## 在测试集上测试并保存结果
```shell
python eval.py --network UNet --backbone resnet50 --model  --result_dir result --test
```
## 数据存放
数据存放目录结构如下：
```text
SpineData
├── test
│   └── image
│       ├── Case196.nii.gz
│       ├── Case201.nii.gz
│       └── ...
├── train
│   ├── groundtruth
│   │   ├── mask_case100.nii.gz
│   │   ├── mask_case101.nii.gz
│   │   ├── ...
│   └── image
│       ├── Case196.nii.gz
│       ├── Case201.nii.gz
│       └── ...
├── val
│   ├── groundtruth
│   │   ├── mask_case100.nii.gz
│   │   ├── mask_case101.nii.gz
│   │   ├── ...
│   └── image
│       ├── Case196.nii.gz
│       ├── Case201.nii.gz
│       └── ...
```
