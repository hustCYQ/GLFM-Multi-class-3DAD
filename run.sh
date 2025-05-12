#!/bin/bash
### 指定该作业需要多少个节点（申请多个节点的话需要您的程序支持分布式计算），必选项
#SBATCH --nodes=1

### 指定该作业在哪个分区队列上执行，gpu作业直接指定gpu分区即可，必选项
#SBATCH --partition=gpu

### 指定每个节点使用的GPU卡数量
### 强磁一号集群一个gpu节点最多可申请使用2张A100卡
### 引力一号集群一个gpu节点最多可申请使用8张A100卡
#SBATCH --gres=gpu:1

### 指定该作业每个计算节点运行多少个任务进程，必选项
#SBATCH --ntasks-per-node=1

### 指定每个任务进程需要的cpu核数(用于多线程任务)，若未指定则默认为每个任务进程分配1个处理器核心，可选项
#SBATCH --cpus-per-task=4


### 指定作业最长运行时间 dd-hh:mm:ss
#SBATCH --time=7-00:00:00

### 指定从哪个项目扣费（需要修改为自己参与的项目名），如果没有这条参数，则从个人账户扣费
#SBATCH --comment=Test1

### 执行您的程序批处理命令，例如：
source ~/.bashrc  #加载环境变量
nvidia-smi
cd /home/ud202380215/GLFM
# python main.py --dataset mvtec --task Single-Class --k_class 1
# python main.py --dataset real --task Single-Class --k_class 1
# python main.py --dataset mvtec --task Multi-Class --k_class 10
python main.py --dataset real --task Multi-Class --k_class 3