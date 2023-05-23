import argparse
import logging
import os
import shutil
import sys
import time
import ipdb
import io
import gin
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorboard.backend.event_processing import event_accumulator, tag_types
import datetime, pytz
import pandas as pd
from tqdm import tqdm

import src.util.gin_utils as gin_utils
from src.aux.dummy_extractor import DummyFeatureExtractor
from src.aux.munk_extractor import MunkNet
from src.aux.network import OFENet
from src.policy import DDPG
from src.policy import PPO
from src.policy import SAC
from src.policy import TD3, TD3_linear
from src.util import misc
from src.util import replay
from src.util.misc import get_target_dim, make_ofe_name, get_default_steps
import trfl.target_update_ops as target_update
from arguments import parse_args


misc.set_gpu_device_growth()
dir_of_env = {'HalfCheetah-v2': 'hc', 
            'Hopper-v2': 'hopper',  
            'Walker2d-v2': 'walker', 
            'Ant-v2': 'ant', 
            'Swimmer-v2': 'swim', 
            'Humanoid-v2': 'human',
            "InvertedDoublePendulum-v2": 'IDPen'}
# Legacy aliases
COMPRESSED_HISTOGRAMS = tag_types.COMPRESSED_HISTOGRAMS
HISTOGRAMS = tag_types.HISTOGRAMS
IMAGES = tag_types.IMAGES
AUDIO = tag_types.AUDIO
SCALARS = tag_types.SCALARS
TENSORS = tag_types.TENSORS
GRAPH = tag_types.GRAPH
META_GRAPH = tag_types.META_GRAPH
RUN_METADATA = tag_types.RUN_METADATA

SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 500,
    IMAGES: 4,
    AUDIO: 4,
    SCALARS: 10000,
    HISTOGRAMS: 1,
    TENSORS: 0,
}

SIZE_GUIDANCE2 = {
    COMPRESSED_HISTOGRAMS: 500,
    IMAGES: 4,
    AUDIO: 4,
    SCALARS: 10000,
    HISTOGRAMS: 1,
    TENSORS: 100,
}

def get_update_every(env_name):
    TARGET_DIM_DICT = {
        "Ant-v2": 150,
        "HalfCheetah-v2": 5,
        "Walker2d-v2": 2,
        "Hopper-v2": 150,
        "Reacher-v2": 11,
        "Humanoid-v2": 1,
        "Swimmer-v2": 200,
        "InvertedDoublePendulum-v2": 11
    }
    return TARGET_DIM_DICT[env_name]

def make_exp_name(args):
    if args.gin is not None:
        extractor_name = gin.query_parameter("feature_extractor.name")

        if extractor_name == "OFE":
            ofe_unit = gin.query_parameter("OFENet.total_units")
            ofe_layer = gin.query_parameter("OFENet.num_layers")
            ofe_act = gin.query_parameter("OFENet.activation")
            ofe_block = gin.query_parameter("OFENet.block")
            ofe_act = str(ofe_act).split(".")[-1]

            ofe_name = make_ofe_name(ofe_layer, ofe_unit, ofe_act, ofe_block)
        elif extractor_name == "Munk":
            munk_size = gin.query_parameter("MunkNet.internal_states")
            ofe_name = "Munk_{}".format(munk_size)
        else:
            raise ValueError("invalid extractor name {}".format(extractor_name))
    else:
        ofe_name = "raw"

    env_name = args.env.split("-")[0]
    exp_name = "{}_{}_{}".format(env_name, args.policy, ofe_name)

    if args.gin is not None:
        exp_name = exp_name + "_tau" + str(args.tau) + "_freq" + str(args.target_update_freq)
        if args.policy == 'PPO':
            exp_name += "_up" + str(args.update_every)

        if args.fourier_type is not None:
            exp_name = exp_name + "_" + args.fourier_type + "_D" + str(args.dim_discretize)

        if args.use_projection == True:
            exp_name = exp_name + "_P" + str(args.projection_dim)

        if args.cosine_similarity == True:
            exp_name = exp_name + "_CosineLoss"
        else:
            exp_name = exp_name + "_L2Loss"

    if args.name is not None:
        exp_name = exp_name + "_" + args.name

    exp_name = exp_name + "_low15-high15-freq-loss"
    # now = datetime.datetime.now(pytz.timezone('Asia/Shanghai'))
    # exp_name = exp_name + "_" + now.strftime("%Y%m%d-%H%M")

    return exp_name

def make_exp_name_ofe(args):
    if args.gin is not None:
        extractor_name = gin.query_parameter("feature_extractor.name")

        if extractor_name == "OFE":
            ofe_unit = gin.query_parameter("OFENet.total_units")
            ofe_layer = gin.query_parameter("OFENet.num_layers")
            ofe_act = gin.query_parameter("OFENet.activation")
            ofe_block = gin.query_parameter("OFENet.block")
            ofe_act = str(ofe_act).split(".")[-1]

            ofe_name = make_ofe_name(ofe_layer, ofe_unit, ofe_act, ofe_block)
        elif extractor_name == "Munk":
            munk_size = gin.query_parameter("MunkNet.internal_states")
            ofe_name = "Munk_{}".format(munk_size)
        else:
            raise ValueError("invalid extractor name {}".format(extractor_name))

        ofe_name = 'ofePaper_' + ofe_name
    else:
        ofe_name = "raw"

    env_name = args.env.split("-")[0]
    exp_name = "{}_{}_{}".format(env_name, args.policy, ofe_name)

    if args.name is not None:
        exp_name = exp_name + "_" + args.name

    if args.policy == 'PPO' and args.gin is not None:
        exp_name += "_up" + str(args.update_every)

    return exp_name


def make_output_dir(dir_root, exp_name, env_name, seed):
    seed_name = "seed{}".format(seed)

    dir_log = os.path.join(dir_root, "log_{}".format(dir_of_env[env_name]), exp_name, seed_name)

    return dir_log

@gin.configurable
def feature_extractor(env_name, dim_state, dim_action, name=None, skip_action_branch=False):
    logger = logging.getLogger(name="main")
    logger.info("Use Extractor {}".format(name))

    if name == "OFE":
        if env_name == "Humanoid-v2":
            target_dim = get_target_dim(env_name)
        else:
            target_dim = dim_state
        extractor = OFENet(dim_state=dim_state, dim_action=dim_action,
                           dim_output=target_dim, skip_action_branch=skip_action_branch)
    elif name == "Munk":
        extractor = MunkNet(dim_state=dim_state, dim_action=dim_action)
    else:
        extractor = DummyFeatureExtractor(dim_state=dim_state, dim_action=dim_action)

    return extractor

if __name__ == "__main__":
    args = parse_args()
    # env_list = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Ant-v2', 'Swimmer-v2', 'Humanoid-v2']
    # env_list = ['HalfCheetah-v2', 'Walker2d-v2', 'Ant-v2', 'Swimmer-v2']
    env_list = ['Ant-v2']
    # policy_list = ['SAC']
    policy_list = ['SAC', 'PPO']
    seed_list = [0,1,2,5,6,7,8,10,11,12]
    # seed_list = [7,8]
    aux_list =  ['raw', 'OFE', 'FSP'] # ['raw'] ['raw', 'OFE', 'FSP']
    
    for env in env_list:
        seed_list = [0,1,2,5,6,7,8,10,11,12]
        args.update_every = get_update_every(env)
        for policy in policy_list:
            args.env = env
            args.policy = policy
            args.dir_root = "./output_{}_img".format(args.policy)
            if args.aux != 'raw':
                args.gin = './gins_{}/{}.gin'.format(policy, env.split("-")[0])
            # CONSTANTS
            if args.gin is not None:
                gin.parse_config_file(args.gin)
            if args.aux == 'FSP':
                exp_name = make_exp_name(args)
            else:
                exp_name = make_exp_name_ofe(args)
            data_dir_log0 = os.path.join(args.dir_root, "log_{}".format(dir_of_env[env]), exp_name)
            seed_list = [int(x.lstrip('seed')) for x in os.listdir(data_dir_log0)]
            seed_list.sort()

            for seed in seed_list:
                
                data_dir_log = make_output_dir(dir_root=args.dir_root, exp_name=exp_name, env_name=env, seed=seed)
                csv_save_dir = os.path.join('results/data/exp_main', args.policy + '-' + args.aux +  '-' + args.env)
                csv_save_dir = os.path.join(csv_save_dir, args.policy + '-' + args.aux +  '-' + args.env + '-s' + str(seed))
                if os.path.exists(csv_save_dir):
                    raise ValueError("output directory {} exists".format(csv_save_dir))
                os.makedirs(csv_save_dir)

                event_data = event_accumulator.EventAccumulator(data_dir_log, size_guidance=SIZE_GUIDANCE)  # a python interface for loading Event data
                event_data.Reload()  # synchronously loads all of the data written so far b
                # print(event_data.Tags())  # print all tags
                keys = event_data.tensors.Keys()  # get all tags,save in a list
                df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
                for key in keys:
                    # print('key={}\n'.format(key))
                    if key.startswith('performance/evaluate_return'):  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
                        df = pd.DataFrame(event_data.Tensors(key))
                        df['Value'] = df['tensor_proto'].map(lambda x: tf.make_ndarray(x))
                        if policy == 'SAC':
                            if args.aux == 'raw' or args.aux == 'FSP':
                                if env == 'Humanoid-v2':
                                    if seed == 10 or seed == 11 or seed == 12:
                                        df['step'] = df['step'].map(lambda x: x+2000)
                                elif env == 'Hopper-v2' and args.aux == 'FSP':
                                    if seed == 10 or seed == 11 or seed == 12:
                                        df['step'] = df['step'].map(lambda x: x+2000)
                                elif env == 'Hopper-v2' and args.aux == 'raw':
                                    if seed == 0 or seed == 10 or seed == 11 or seed == 12:
                                        df['step'] = df['step'].map(lambda x: x+2000)
                        if env == 'Humanoid-v2' and args.aux == 'raw':
                            df = df[:600]
                        if env == 'Humanoid-v2' and args.aux == 'OFE':
                            df = df[:600]
                        if env == 'Ant-v2' and args.policy == 'SAC':
                            df = df[:598]
                        if env == 'Ant-v2' and args.policy == 'PPO':
                            df = df[:600]
                        print('export key {} into data'.format(key))

                df.to_csv(os.path.join(csv_save_dir, 'progress.csv'))

                print("Tensorboard data exported from [{}] into dir [{}] successfully".format(data_dir_log, csv_save_dir))


