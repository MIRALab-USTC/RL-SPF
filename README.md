# SPF
State Sequences Prediction via Fourier Transform (SPF) is a representation learning method that predicts the Fourier transform of state sequences to extract the underlying structural information in state sequences for learning expressive representations efficiently. 
It can be combined with algorithms such as PPO and SAC.

This repository contains SPF implementation, OFENet implementation, RL algorithms, and hyperparameters, which we used in our paper. 

The implementation of OFENet is follow the paper [Can Increasing Input Dimensionality Improve Deep Reinforcement Learning?](https://arxiv.org/abs/2003.01629) from code link http://www.merl.com/research/license/OFENet.


## Requirement
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


### MuJoCo

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

## Examples
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


## Retrieve results

`eager_main*.py` generates a log file under "output_algo" directory. 
You can watch the result of an experiment with tensorboard.

```bash
$ tensorboard --logdir ./output_SAC/log_env
```