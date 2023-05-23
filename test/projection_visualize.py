import argparse
import copy
import logging
import os
import shutil
import seaborn as sns
import sys
import time
import ipdb
import io

import gin
import gym
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from scipy.fftpack import fft,ifft

import teflon.util.gin_utils as gin_utils
from arguments import parse_args
from teflon.ofe.dummy_extractor import DummyFeatureExtractor
from teflon.ofe.munk_extractor import MunkNet
from teflon.ofe.network import OFENet
from teflon.ofe.network_ofePaper import OFENet as OFENet_ofePaper
from teflon.policy import DDPG
from teflon.policy import SAC
from teflon.policy import TD3
# from teflon.fourier.network import dtftObs, dftObs
from teflon.util import misc
from teflon.util import replay
from teflon.util.misc import get_target_dim, make_ofe_name
import trfl.target_update_ops as target_update
from teflon.transfer.task import *

misc.set_gpu_device_growth()
dir_of_env = {'HalfCheetah-v2': 'hc', 'HalfCheetah_noise-v2': 'hcNoise', 'HalfCheetah_transfer-v2': 'hcTrans', 
            'Hopper-v2': 'hopper',  'HopperP-v2': 'hopperP', 'HopperV-v2': 'hopperV', 'Hopper_noise-v2': 'hopperNoise', 
            'Walker2d-v2': 'walker', 'Walker2dP-v2': 'walkerP',  'Walker2dV-v2': 'walkerV', 'Walker2d_noise-v2': 'walkerNoise', 
            'Ant-v2': 'ant', 'AntP-v2': 'antP', 'AntV-v2': 'antV', 'Ant_noise-v2': 'antNoise', 
            'Swimmer-v2': 'swim', 
            'Humanoid-v2': 'human'}

# bash run.sh
# drawing true and pred fourier after projection

def make_env(env_name, test=True):
    if env_name == "Hopper":

        env = gym.make("Hopper-v2")
        eval_env = gym.make("Hopper-v2")

    elif env_name == "HopperP":

        env = HopperP()
        eval_env = HopperP()

    elif env_name == "HopperV":

        env = HopperV()
        eval_env = HopperV()

    elif env_name == "Hopper_noise":

        env = Hopper_noise()
        eval_env = Hopper_noise()

    elif env_name == "Walker2d":

        env = gym.make("Walker2d-v2")
        eval_env = gym.make("Walker2d-v2")

    elif env_name == "Walker2dV":

        env = Walker2dV()
        eval_env = Walker2dV()

    elif env_name == "Walker2dP":

        env = Walker2dP()
        eval_env = Walker2dP()

    elif env_name == "Walker2d_noise":
        env = Walker2d_noise()
        eval_env = Walker2d_noise()

    elif env_name == "Ant":

        env = gym.make("Ant-v2")
        eval_env = gym.make("Ant-v2")

    elif env_name == "AntV":

        env = AntV()
        eval_env = AntV()

    elif env_name == "AntP":

        env = AntP()
        eval_env = AntP()

    elif env_name == "Ant_noise":

        env = Ant_noise()
        eval_env = Ant_noise()

    elif env_name == "HalfCheetah":

        env = gym.make("HalfCheetah-v2")
        eval_env = gym.make("HalfCheetah-v2")

    elif env_name == "HalfCheetah_noise":

        env = HalfCheetah_noise()
        eval_env = HalfCheetah_noise()
    
    elif env_name == "HalfCheetah_transfer":

        env = gym.make("HalfCheetah-v2")
        eval_env = gym.make("HalfCheetah_transfer-v2")

    elif env_name == "Swimmer":

        env = gym.make("Swimmer-v2")
        eval_env = gym.make("Swimmer-v2")

    elif env_name == "Humanoid":

        env = gym.make("Humanoid-v2")
        eval_env = gym.make("Humanoid-v2")

    if test == True:
        return env, eval_env
    else:
        return env

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


def draw_tsne(feature, group = 6):
    tsne = TSNE(n_components=2).fit_transform(feature)
    batch_size = feature.shape[0] // group

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['red', 'blue', 'green', 'brown', 'yellow', 'cyan']

    fig, ax = plt.subplots()
    for idx, c in enumerate(colors[:group]):
        indices = np.arange(idx*batch_size, (idx+1)*batch_size, dtype=np.int32)
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        ax.scatter(current_tx, current_ty, c=c, label=idx)

    legends = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Ant-v2', 'Swimmer-v2', 'Humanoid-v2']
    ax.legend(legends)
    ax.set_title('t-sne of different tasks\' projections')

    img_save_dir = './results/img/periodicity'
    # if not os.path.exists(os.path.join(img_save_dir, 'tnse')):
    #     os.mkdir(os.path.join(img_save_dir, 'tsne'))

    plt.savefig(os.path.join(img_save_dir, 'tsne'), dpi=300)
    plt.close ('all')  

    return

def evaluate_policy(env, policy, eval_episodes=10, record_state=False):
    avg_reward = 0.
    episode_length = []
    state_list = []
    if record_state==True:
        state_list = np.zeros([env._max_episode_steps, env.observation_space.shape[0]], dtype=np.float32)

    for i in range(eval_episodes):
        state = env.reset()
        cur_length = 0

        done = False
        while not done:
            if record_state == True and i==0:
                state_list[cur_length,:] = state
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            cur_length += 1

        episode_length.append(cur_length)

    avg_reward /= eval_episodes
    avg_length = np.average(episode_length)
    return avg_reward, avg_length, state_list


def make_exp_name(args):
    if args.gin is not None:
        extractor_name = gin.query_parameter("feature_extractor.name")

        if extractor_name == "OFE":
            ofe_unit = gin.query_parameter("OFENet.total_units")
            ofe_layer = gin.query_parameter("OFENet.num_layers")
            ofe_act = gin.query_parameter("OFENet.activation")
            ofe_act = str(ofe_act).split(".")[-1]

            ofe_name = make_ofe_name(ofe_layer, ofe_unit, ofe_act, args.block)

        elif extractor_name == "Munk":
            munk_size = gin.query_parameter("MunkNet.internal_states")
            ofe_name = "Munk_{}".format(munk_size)
        else:
            raise ValueError("invalid extractor name {}".format(extractor_name))
    else:
        ofe_name = "raw"

    if args.original == True:
        ofe_name = "raw"

    env_name = args.env.split("-")[0]
    exp_name = "{}_{}_{}".format(env_name, args.policy, ofe_name)

    if args.gin is not None and args.original == False:
        exp_name = exp_name + "_tau" + str(args.tau) + "_freq" + str(args.target_update_freq)

        if args.fourier_type is not None:
            exp_name = exp_name + "_" + args.fourier_type + "_D" + str(args.dim_discretize)

        if args.use_projection == True:
            exp_name = exp_name + "_P" + str(args.projection_dim)

    if args.name is not None:
        exp_name = exp_name + "_" + args.name
    
    # exp_name = exp_name + "_" + time.strftime("%Y%m%d-%H%M")

    return exp_name


def make_policy(policy, env_name, extractor=None, extractor_target=None, units=256, original=False):
    env = make_env(env_name.split("-")[0], test=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    n_units = [units, units]

    if policy == "SAC":
        scale_reward = SAC.get_default_scale_reward(env_name)
        policy = SAC.SAC(state_dim, action_dim, max_action, 
                        feature_extractor=extractor, feature_extractor_target=extractor_target, 
                        scale_reward=scale_reward,
                        actor_units=n_units, q_units=n_units, v_units=n_units)
        print("We use SAC with extractor!")

    elif policy == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action, feature_extractor=extractor)
    elif policy == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(400, 300), feature_extractor=extractor)
    elif policy == "TD3small":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(256, 256), feature_extractor=extractor)
    else:
        raise ValueError("invalid policy {}".format(policy))

    return policy


def make_policy_ofePaper(policy, env_name, extractor, units=256):
    env = make_env(env_name.split("-")[0], test=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    n_units = [units, units]

    if policy == "SAC":
        scale_reward = SAC.get_default_scale_reward(env_name)
        policy = SAC.SAC(state_dim, action_dim, max_action, feature_extractor=extractor, scale_reward=scale_reward,
                         actor_units=n_units, q_units=n_units, v_units=n_units)
    elif policy == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action, feature_extractor=extractor)
    elif policy == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(400, 300), feature_extractor=extractor)
    elif policy == "TD3small":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(256, 256), feature_extractor=extractor)
    else:
        raise ValueError("invalid policy {}".format(policy))

    return policy


def make_output_dir(dir_root, exp_name, env_name, seed, ignore_errors):
    dir_logs = []

    seed_name = "seed{}".format(seed)
    dir_log = os.path.join(dir_root, "log_{}".format(dir_of_env[env_name]), exp_name, seed_name)
    model_save_dir = os.path.join(dir_log, 'model')
    dir_logs.append(dir_log)

    return dir_logs, model_save_dir


@gin.configurable
# state representation
def feature_extractor(extractor_kwargs=dict(), name=None, skip_action_branch=False):
    logger = logging.getLogger(name="main")
    logger.info("Use Extractor {}".format(name))

    extractor = OFENet(**extractor_kwargs, skip_action_branch=skip_action_branch)
    extractor_target = OFENet(**extractor_kwargs, skip_action_branch=skip_action_branch)
    target_update.update_target_variables(extractor_target.weights, extractor.weights)
    # print network parameters of OFENet
    print("OFENet's network structure:\n")
    tvars = extractor.trainable_variables
    for var in tvars:
        print(" name = %s, shape = %s" % (var.name, var.shape))

    return extractor, extractor_target


def feature_extractor_ofePaper(env_name, dim_state, dim_action, name=None, skip_action_branch=False):
    logger = logging.getLogger(name="main")
    logger.info("Use Extractor {}".format(name))

    if env_name == "Humanoid-v2":
        target_dim = get_target_dim(env_name)
    else:
        target_dim = dim_state
    extractor = OFENet_ofePaper(dim_state=dim_state, dim_action=dim_action,
                        dim_output=target_dim, skip_action_branch=skip_action_branch)

    # print network parameters of OFENet
    print("OFENet's network structure:\n")
    tvars = extractor.trainable_variables
    for var in tvars:
        print(" name = %s, shape = %s" % (var.name, var.shape))

    return extractor

def main():
    logger = logging.Logger(name="main")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s',
                                           datefmt="%m/%d %I:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    args = parse_args()
    args.random_collect = 500
    args_text = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logger.info(args_text)

    # make sure the normal execution of replay_buffer.sample_with_Hth_states
    assert args.random_collect - args.dim_discretize > args.batch_size

    if args.eval_freq % args.summary_freq != 0:
        logger.error("eval_freq must be divisible by summary_freq.")
        sys.exit(-1)

    env_names = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Ant-v2', 'Swimmer-v2', 'Humanoid-v2']
    # env_names = ['HalfCheetah-v2']
    projs_for_all_tasks = np.array([])
    for env_name in env_names:
        # CONSTANTS
        args.gin = os.path.join('./gins_ofePaper', env_name.split('-')[0] + '.gin')
        if args.gin is not None:
            gin.parse_config_file(args.gin)

        args.env = env_name
        policy_name = args.policy
        batch_size = args.batch_size
        dir_root = args.dir_root
        seed = args.seed

        exp_name = make_exp_name(args)
        seed = 10

        env, eval_env = make_env(env_name.split("-")[0])

        # Set seeds
        env.seed(seed)
        eval_env.seed(seed + 1000)
        # tf.set_random_seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]

        if args.original == False:
            extractor_kwargs = {
                "dim_state": dim_state, 
                "dim_action": dim_action, 
                "dim_output": get_target_dim(env_name),
                "dim_discretize": args.dim_discretize, 
                "block": args.block,
                "fourier_type": args.fourier_type, 
                "discount": args.discount, 
                "use_projection": args.use_projection, 
                "projection_dim": args.projection_dim, 
                "cosine_similarity": args.cosine_similarity,
            }
            extractor, extractor_target = feature_extractor(extractor_kwargs=extractor_kwargs)
        else:
            extractor = None
            extractor_target = None

        # Initialize replay buffer
        if args.fourier_type == 'dtft':
            replay_buffer = replay.ReplayBuffer(state_dim=dim_state, action_dim=dim_action, capacity=1000000)
        elif args.fourier_type == 'dft':
            replay_buffer = replay.LongReplayBuffer(state_dim=dim_state, action_dim=dim_action, horizon = args.dim_discretize, capacity=1000000)
        else:
            raise ValueError("Invalid parameter fourier_type {}".format(args.fourier_type))

        total_timesteps = np.array(0, dtype=np.int32)
        episode_timesteps = 0
        episode_return = 0
        state = env.reset()

        logger.info("collecting random {} transitions".format(args.random_collect))

        print("Initialization: I am collecting samples randomly!")
        for i in range(args.random_collect):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            episode_timesteps += 1
            total_timesteps += 1

            done_flag = done
            if episode_timesteps == env._max_episode_steps:
                done_flag = False

            replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done_flag)
            state = next_state

            if done:
                state = env.reset()
                episode_timesteps = 0
                episode_return = 0

        # pretraining the extractor, extractor's trainable variables are incomplete until the extractor has been trained once.
        if args.original == False:
            print("Pretrain: I am pretraining the extractor!")
            for i in range(2):

                tf.summary.experimental.set_step(i - args.pre_train_step)

                if args.fourier_type == 'dtft':

                    sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones = replay_buffer.sample(
                        batch_size=batch_size)
                    sample_next_actions = [env.action_space.sample() for k in range(batch_size)]
                    
                    pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred = extractor.train(extractor_target, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)
                    
                elif args.fourier_type == "dft":
                    sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones, sample_Hth_states = replay_buffer.sample_with_Hth_states(
                        batch_size=batch_size)
                    sample_next_actions = [env.action_space.sample() for k in range(batch_size)]
                    
                    pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred = extractor.train(extractor_target, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones, sample_Hth_states)

         
            tvars = extractor.projection.trainable_variables
            for var in tvars:
                print(" name = %s, shape = %s" % (var.name, var.shape))
    
        # test model in environme nts with different masses and frictions
        algo = ''
        img_save_dir = './results/img/periodicity'
        if not os.path.exists(os.path.join(img_save_dir, dir_of_env[env_name])):
            os.mkdir(os.path.join(img_save_dir, dir_of_env[env_name]))

        _, model_save_dir = make_output_dir(dir_root=dir_root, exp_name=exp_name, env_name=env_name, seed=seed,
                                             ignore_errors=args.force)
        
        _, eval_env = make_env(env_name.split("-")[0])
        eval_env.seed(seed + 1000)
        
        if 'ofePaper' in model_save_dir:
            algo = 'ofePaper'
            # Load extractor
            extractor_ofePaper = feature_extractor_ofePaper(env_name, dim_state, dim_action)
            extractor_ofePaper.load_weights(os.path.join(model_save_dir,'extractor_model'))
            logger.info("Feature extractor's model has been loaded.")
            # Load policy
            policy = make_policy_ofePaper(policy=policy_name, 
                                    env_name=env_name, 
                                    extractor=extractor_ofePaper, 
                                    units=args.sac_units)
            policy.load(model_save_dir)
            logger.info("Policy's model has been loaded.")
        
        elif 'raw' in model_save_dir:
            algo = 'raw'
            # Load policy
            policy = make_policy(policy=policy_name, 
                                env_name=env_name, 
                                extractor=None, 
                                extractor_target=None, 
                                units=args.sac_units, 
                                original=True)
            policy.load(model_save_dir)
            logger.info("Policy's model has been loaded.")

        else:     
            algo = 'FoSta'
            # Load extractor
            extractor.load_weights(os.path.join(model_save_dir,'extractor_model'))
            extractor_target.load_weights(os.path.join(model_save_dir,'extractor_target_model'))
            logger.info("Feature extractor's model has been loaded.")

            # Load policy
            policy = make_policy(policy=policy_name, 
                                env_name=env_name, 
                                extractor=extractor, 
                                extractor_target=extractor_target, 
                                units=args.sac_units, 
                                original=args.original)
            policy.load(model_save_dir)
            logger.info("Policy's model has been loaded.")
        
        # # print parameters of extractor
        # np.set_printoptions(threshold=np.inf)
        # file = open(os.path.join(img_save_dir, dir_of_env[env_name], 'extractor_weights.txt'), 'w')
        # for v in extractor.trainable_variables:
        #     file.write(str(v.name) + '\n')
        #     file.write(str(v.shape) + '\n')
        #     file.write(str(v.numpy()) + '\n')
        # file.close()
        
        # test model
        try:
            exp_data = pd.read_csv(os.path.join(img_save_dir, dir_of_env[env_name], '{}_state_action_data.csv'.format(dir_of_env[env_name])))  # pd.read_csv读取以逗号为分割符的文件
        except:
            print('Could not read from %s'%os.path.join(img_save_dir, dir_of_env[env_name], '{}_state_action_data.csv'.format(dir_of_env[env_name])))
            continue

        exp_data_len = len(exp_data)
        horizon = 200
        s_future_data = tf.cast(np.array(exp_data.iloc[exp_data_len - horizon + 1: exp_data_len, 1: dim_state + 1]), dtype=tf.float32)
        # input = exp_data.iloc[exp_data_len - horizon].values.tolist()
        input_s = np.array(exp_data.iloc[[exp_data_len - horizon], 1: dim_state + 1])
        input_a = np.array(exp_data.iloc[[exp_data_len - horizon], dim_state + 1: dim_state + dim_action + 1])

        with tf.device("/gpu:{}".format(0)):
            with tf.GradientTape(persistent=True) as tape:
                fourier_pred_re, fourier_pred_im = extractor([input_s, input_a], training=False)

                con_gamma = tf.cast(np.diag([pow(args.discount, k) for k in range(horizon - 1)]), dtype=tf.float32)
                ratio = 2 * np.pi / args.dim_discretize
                con = tf.matmul(tf.cast(ratio * np.expand_dims(np.arange(args.dim_discretize), axis=1), dtype=tf.float32), \
                                tf.cast(np.expand_dims(np.arange(horizon - 1), axis=0), dtype=tf.float32))
    
                con_re = tf.math.cos(con)
                con_im = - tf.math.sin(con)
                fourier_true_re = tf.matmul(tf.matmul(con_re, con_gamma), s_future_data)
                fourier_true_im = tf.matmul(tf.matmul(con_im, con_gamma), s_future_data)
                fourier_true_re = tf.expand_dims(fourier_true_re, axis=0)
                fourier_true_im = tf.expand_dims(fourier_true_im, axis=0)

                fourier_pred_re_proj = extractor.projection(fourier_pred_re, training=False)  #1*256
                fourier_pred_im_proj = extractor.projection(fourier_pred_im, training=False)
                fourier_true_re_proj = extractor.projection(fourier_true_re, training=False)
                fourier_true_im_proj = extractor.projection(fourier_true_im, training=False)
                
                if projs_for_all_tasks.shape[0] == 0:
                    projs_for_all_tasks = np.concatenate((np.array(fourier_true_re_proj).T, np.array(fourier_true_im_proj).T), axis = 1)
                else:
                    a = np.concatenate((np.array(fourier_true_re_proj).T, np.array(fourier_true_im_proj).T), axis = 1)
                    projs_for_all_tasks = np.append(projs_for_all_tasks, a, axis = 0)

                fourier_pred_re_proj = tf.squeeze(fourier_pred_re_proj)  #1*256
                fourier_pred_im_proj = tf.squeeze(fourier_pred_im_proj)
                fourier_true_re_proj = tf.squeeze(fourier_true_re_proj)
                fourier_true_im_proj = tf.squeeze(fourier_true_im_proj)

                loss_fun = tf.keras.losses.CosineSimilarity(axis=-1)
                # fourier_csError_re = loss_fun(fourier_pred_re_proj, fourier_true_re_proj)
                # fourier_csError_im = loss_fun(fourier_pred_im_proj, fourier_true_im_proj)
            
            del tape
        
        types = ['true and predicted', 'csError']
        complex_types = ['real', 'imaginary']  # ['module', 'angle']
        x = np.arange(fourier_pred_re_proj.shape[0])

        for complex_type in complex_types:
            
            for type0 in types:
                
                if not os.path.exists(os.path.join(img_save_dir, dir_of_env[env_name], '{}_fourier_{}'.format(type0, complex_type))):
                    os.mkdir(os.path.join(img_save_dir, dir_of_env[env_name], '{}_fourier_{}'.format(type0, complex_type)))

                fig, ax = plt.subplots()
                logger.info('Drawing the projection of {}\'s fourier {}\n'.format(env_name.split("-")[0], complex_type))

                if complex_type == 'real':
                    if type0 == 'true and predicted':
                        # ax.plot(x, fourier_true_re_proj, linewidth=2, linestyle='solid')
                        # ax.plot(x, fourier_pred_re_proj, linewidth=2, linestyle='dashed')
                        ax.plot(x, fourier_true_re_proj, linewidth=2)
                        ax.plot(x, fourier_pred_re_proj, linewidth=1)
                        ax.legend(['true', 'pred'])

                        # ax.plot(ratio * np.arange(args.dim_discretize), tf.squeeze(fourier_true_re)[:, 0])
                        # ax.plot(ratio * np.arange(args.dim_discretize), tf.squeeze(fourier_pred_im)[:, 0])
                        # ax.legend(['true', 'pred'])

                        # ax.legend(['true_proj', 'pred_proj', 'true', 'pred'])
                        # plt.ylim((0, 0.1))
                    elif type0 == 'csError':
                        print('cosine similarity of the projection of {}\'s fourier {}\n'.format(loss_fun(fourier_pred_re_proj, fourier_true_re_proj), \
                                                                                                    env_name.split("-")[0], complex_type))

                elif complex_type == 'imaginary':
                    if type0 == 'true and predicted':
                        # ax.plot(x, fourier_true_im_proj, linewidth=2, linestyle='solid')
                        # ax.plot(x, fourier_pred_im_proj, linewidth=2, linestyle='dashed')
                        ax.plot(x, fourier_true_im_proj, linewidth=2)
                        ax.plot(x, fourier_pred_im_proj, linewidth=1)
                        ax.legend(['true', 'pred'])
                    elif type0 == 'csError':
                         print('cosine similarity of the projection of {}\'s fourier {}\n'.format(loss_fun(fourier_pred_im_proj, fourier_true_im_proj), \
                                                                                                    env_name.split("-")[0], complex_type))
                
                ax.set_xlabel('Omega')
                ax.set_ylabel('fourier {}'.format(complex_type))
                ax.set_title('the projection of {}\'s fourier {}\n'.format(env_name.split("-")[0], complex_type))

                plt.savefig(os.path.join(img_save_dir, dir_of_env[env_name], '{}_fourier_{}'.format(type0, complex_type), '{}\'s_{}_fourier_{}'.format(dir_of_env[env_name], type0, complex_type)), dpi=300)
                plt.close ('all')
        
        logger.info('Done')

    draw_tsne(projs_for_all_tasks, group = len(env_names))


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )
    main()