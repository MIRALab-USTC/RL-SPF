# OFENet
OFENet is a feature extractor network for low-dimensional data to improve performance of Reinforcement Learning.
It can be combined with algorithms such as PPO, DDPG, TD3, and SAC.

This repository contains OFENet implementation, RL algorithms, and hyperparameters, which
we used in our paper. We ran these codes on Ubuntu 18.04 & GeForce 1060.

## Requirement

```bash
$ conda create -n src python=3.6 anaconda
$ conda activate src
$ conda install cudatoolkit=10.0 cudnn tensorflow-gpu==2.0.0
$ pip install -r requirements.txt
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

To train an agent with OFENet, run the below commands at the project root.

```bash
$ export PYTHONPATH=.
$ python3 src/tool/eager_main.py --policy SAC \
                                  --env HalfCheetah-v2 \
                                  --gin ./gins/HalfCheetah.gin \
                                  --seed 0
```

If you want to combine OFENet with TD3 or DDPG, change the policy like

```bash
$ export PYTHONPATH=.
$ python3 src/tool/eager_main.py --policy TD3 \
                                  --env HalfCheetah-v2 \
                                  --gin ./gins/HalfCheetah.gin \
                                  --seed 0
```

When you want to run an agent in another environment, change the policy and 
the hyperparameter file (.gin).

```bash
$ python3 src/tool/eager_main.py --policy SAC \
                                  --env Walker2d-v2  \
                                  --gin ./gins/Walker2d.gin \
                                  --seed 0
```

When you don't specify a gin file, you train an agent with raw observations. 

```bash
$ python3 src/tool/eager_main.py --policy SAC \
                                  --env HalfCheetah-v2 \
                                  --seed 0
```

ML-SAC is trained with the below command.

```bash
$ python3 src/tool/eager_main.py --policy SAC \
                                  --env HalfCheetah-v2 \
                                  --gin ./gins/Munk.gin \
                                  --seed 0
```

## Retrieve results

`eager_main.py` generates a log file under "log" directory. 
You can watch the result of an experiment with tensorboard.

```bash
$ tensorboard --logdir ./log
```

## Contact

If you have problem running codes, please contact to Kei Ota (dev.ohtakei [at] gmail.com)

In TD3 and Hopper.gin, change num_layer from 6 to 8