# SPF
![](./figures/network%20structure.png)
This is the code of paper "**State Sequences Prediction via Fourier Transform for Representation Learning**". Mingxuan Ye, Yufei Kuang, Jie Wang, Rui Yang, Wengang Zhou, Houqiang Li, Feng Wu. NeurIPS 2023 (Spotlight).

**State Sequences Prediction via Fourier Transform (SPF)** is a representation learning method that predicts the Fourier transform of state sequences to extract the underlying structural information in state sequences for learning expressive representations efficiently. 
It can be combined with algorithms such as PPO and SAC.

This repository contains SPF implementation, OFENet implementation, RL algorithms, and hyperparameters, which we used in our paper. 

The implementation of OFENet is follow the paper [Can Increasing Input Dimensionality Improve Deep Reinforcement Learning?](https://arxiv.org/abs/2003.01629) from code link http://www.merl.com/research/license/OFENet.


## Requirements
### CUDA10.2
We ran these codes on CUDA 10.2 & Driver 440.33.01 & GeForce RTX 2080 Ti.

```bash
$ conda create -n spf python=3.6
$ source activate spf
$ conda install cudatoolkit=10.0 cudnn tensorflow-gpu==2.0.0
$ pip install -r ./requirements/requirements_tf20.txt  #  run this line at the project root
```
If there are issues with the network connection, an offline installation of tensorflow2.0.0 can be attempted. The corresponding dependency packages are as follows:

+ cudatoolkit==10.0：[linux-64/cudatoolkit-10.0.130-0.tar.bz2](https://anaconda.org/anaconda/cudatoolkit/10.0.130/download/linux-64/cudatoolkit-10.0.130-0.tar.bz2)
+ cudnn==7.6.5：[linux-64/cudnn-7.6.5-cuda10.0_0.tar.bz2](https://anaconda.org/anaconda/cudnn/7.6.5/download/linux-64/cudnn-7.6.5-cuda10.0_0.tar.bz2)

Download the offline package to a local directory and then execute the following code at that local directory:
```bash
$ conda install --offline cudatoolkit-10.0.130-0.tar.bz2
$ conda install --offline cudnn-7.6.5-cuda10.0_0.tar.bz2
$ conda install cudatoolkit=10.0 cudnn tensorflow-gpu==2.0.0
```

### CUDA11.8
We have also provided the environment setup on CUDA 11.8 & Driver 520.61.05 & GeForce RTX 3090 Ti.

Create the environment:
```bash
$ conda create -n spf python=3.7
$ source activate spf
```
Install dependencies for Tensorflow 2.6.0:
+ cudatoolkit==11.3：
```bash
$ wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cudatoolkit-11.3.1-h2bc3f7f_2.conda
$ conda install --use-local cudatoolkit-11.3.1-h2bc3f7f_2.conda
```
+ cudnn==8.2.1：
```bash
$ wget https://repo.anaconda.com/pkgs/main/linux-64/cudnn-8.2.1-cuda11.3_0.conda
$ conda install --use-local cudnn-8.2.1-cuda11.3_0.conda
```
Install Tensorflow 2.6.0 and other libraries:
```bash
$ pip install tensorflow==2.6.0
$ pip install tensorflow-gpu==2.6 --user
$ pip install tensorflow-probability==0.14.0
$ pip install -r ./requirements/requirements_tf26.txt  #  run this line at the project root
```

### MuJoCo200

Install MuJoCo 2.0 from the [official web site](http://www.mujoco.org/index.html).

```bash
$ mkdir ~/.mujoco
$ cd ~/.mujoco
$ wget https://www.roboti.us/download/mujoco200_linux.zip
$ unzip mujoco200_linux.zip
$ mv mujoco200_linux mujoco200
$ cp /path/to/mjkey.txt ./
$ pip install mujoco_py
```


## Usage

Go to the root directory `RL-SPF`. Below is an illustration of the directory structure.

```
RL-SPF
├── curves
├── gins (hyperparameters configuration of neural networks)
├── my_log (files for saving terminal outputs)
├── src (core codes)
│   ├── aux
│   │   ├── blocks.py (structure of net blocks)
│   │   ├── network.py  (network and update of SPF, our method)
│   │   ├── network_ofePaper.py (network and update of OFENet, baseline)
│   │   ├── ...
│   ├── policy (classic RL algorithms)
│   │   ├── SAC.py
│   │   ├── PPO.py
│   │   └── ...
│   ├── util
├── tool_for_plot (visualization tools)
├── trfl
├── README.md
├── arguments.py (hyperparameters configuration of SPF)
├── eager_main*.py (train and evaluate)
├── run*.sh (execute commands)
```

### Reproduce the results
Create a folder named "my_log" at the project root before running the code.

To train an agent with SPF combined with SAC, run the below command at the project root. The code then starts training the agent on 6 MuJoCo tasks in seed 0.


```bash
$ bash run_spf_sac_seed.sh
```

If you want to combine OFENet with PPO, change the command like

```bash
$ bash run_spf_ppo_seed.sh
```

When you want to train an agent with OFENet combined with SAC and the original SAC algorithm, run the below command at the project root.
```bash
$ bash run_ofenet_and_raw_sac_seed.sh
```

If you want to train an agent with OFENet combined with PPO and the original PPO algorithm, change the command like

```bash
$ bash run_ofenet_and_raw_ppo_seed.sh
```


### Retrieve the results

`eager_main*.py` generates a log file under `./output_${algo}/log_${env}` directory. 
You can watch the result of an experiment with tensorboard.

```bash
$ tensorboard --logdir ./output_SAC/log_hc
```