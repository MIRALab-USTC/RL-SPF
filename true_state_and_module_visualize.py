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
from teflon.ofe.network_ofePaper import OFENet_ofePaper
from teflon.policy import DDPG
from teflon.policy import SAC
from teflon.policy import TD3
# from teflon.fourier.network import dtftObs, dftObs
from teflon.util import misc
from teflon.util import replay
from teflon.util.misc import get_target_dim, make_ofe_name
import trfl.target_update_ops as target_update

misc.set_gpu_device_growth()
dir_of_env = {'HalfCheetah-v2': 'hc', 
            'Hopper-v2': 'hopper',  
            'Walker2d-v2': 'walker', 
            'Ant-v2': 'ant', 
            'Swimmer-v2': 'swim', 
            'Humanoid-v2': 'human'}

# bash test/run_test.sh
# drawing state sequence and true fourier's module

def evaluate_policy(env, policy, eval_episodes=10, record_state=False):
    avg_reward = 0.
    episode_length = []
    state_list = []
    if record_state==True:
        state_list = np.zeros([env._max_episode_steps, env.observation_space.shape[0]], dtype=np.float32)
        action_list = np.zeros([env._max_episode_steps, env.action_space.shape[0]], dtype=np.float32)

    for i in range(eval_episodes):
        state = env.reset()
        cur_length = 0

        done = False
        while not done:
            action = policy.select_action(np.array(state))
            if record_state == True and i==0:
                state_list[cur_length,:] = state
                action_list[cur_length,:] = action
            
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            cur_length += 1

        episode_length.append(cur_length)

    avg_reward /= eval_episodes
    avg_length = np.average(episode_length)
    return avg_reward, avg_length, state_list, action_list


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
    
    exp_name += "_low15-high15-freq-loss"

    return exp_name

def make_exp_name_ofePaper(args):
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


    return exp_name

def make_policy(policy, env_name, extractor, units=256):
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    n_units = [units, units]

    if policy == "SAC":
        scale_reward = SAC.get_default_scale_reward(env_name)
        policy = SAC.SAC(state_dim, action_dim, max_action, 
                        feature_extractor=extractor,
                        scale_reward=scale_reward,
                        actor_units=n_units, q_units=n_units, v_units=n_units)
        print("We use SAC algorithm!")
    elif policy == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action, feature_extractor=extractor)
    elif policy == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(400, 300), feature_extractor=extractor)
        print("We use TD3 algorithm!")
    elif policy == "TD3small":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(256, 256), feature_extractor=extractor)
    else:
        raise ValueError("invalid policy {}".format(policy))

    return policy

def make_output_dir(dir_root, exp_name, env_name, seed, ignore_errors):
    seed_name = "seed{}".format(seed)

    dir_log = os.path.join(dir_root, "log_{}".format(dir_of_env[env_name]), exp_name, seed_name)

    for cur_dir in [dir_log]:  # 如果存在文件名一样的，则递归的删除目录
        if not os.path.exists(cur_dir):
            raise ValueError("output directory {} does not exist".format(cur_dir))

    return dir_log

@gin.configurable
def feature_extractor(extractor_kwargs=dict(), name=None, skip_action_branch=False):
    logger = logging.getLogger(name="main")
    logger.info("Use Extractor {}".format(name))
    extractor_target = None

    if name == "OFE":
        extractor = OFENet(**extractor_kwargs, skip_action_branch=skip_action_branch)
        extractor_target = OFENet(**extractor_kwargs, skip_action_branch=skip_action_branch)
        target_update.update_target_variables(extractor_target.weights, extractor.weights)
        # print network parameters of OFENet
        print("OFENet's network structure:\n")
        tvars = extractor.trainable_variables
        for var in tvars:
            print(" name = %s, shape = %s" % (var.name, var.shape))

    elif name == "Munk":
        extractor = MunkNet(dim_state=dim_state, dim_action=dim_action)
    else:
        extractor = DummyFeatureExtractor(dim_state=dim_state, dim_action=dim_action)

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

    # env_names = ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2', 'Ant-v2', 'Swimmer-v2', 'Humanoid-v2']
    env_names = [args.env]

    for env_name in env_names:
        # CONSTANTS
        args.gin = os.path.join('./gins_test', env_name.split('-')[0] + '.gin')
        if args.gin is not None:
            gin.parse_config_file(args.gin)

        args.env = env_name
        policy_name = args.policy
        batch_size = args.batch_size
        dir_root = args.dir_root
        seed = args.seed

        # exp_name = make_exp_name(args)
        exp_name = make_exp_name_ofePaper(args)

        env = gym.make(env_name)
        eval_env = gym.make(env_name)

        # Set seeds
        env.seed(seed)
        eval_env.seed(seed + 1000)
        # tf.set_random_seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]

        extractor_kwargs = {
            "dim_state": dim_state, 
            "dim_action": dim_action, 
            "dim_output": get_target_dim(env_name),
            "dim_discretize": args.dim_discretize, 
            "fourier_type": args.fourier_type, 
            "discount": args.discount, 
            "use_projection": args.use_projection, 
            "projection_dim": args.projection_dim, 
            "cosine_similarity": args.cosine_similarity,
        }
        extractor, extractor_target = feature_extractor(extractor_kwargs=extractor_kwargs)

        # Initialize replay buffer
        replay_buffer = replay.ReplayBuffer(state_dim=dim_state, action_dim=dim_action, capacity=1000000)

        total_timesteps = np.array(0, dtype=np.int32)
        episode_timesteps = 0
        episode_return = 0
        state = env.reset()

        logger.info("collecting random {} transitions".format(args.random_collect))

        # print("Initialization: I am collecting samples randomly!")
        # for i in range(args.random_collect):
        #     action = env.action_space.sample()
        #     next_state, reward, done, _ = env.step(action)

        #     episode_return += reward
        #     episode_timesteps += 1
        #     total_timesteps += 1

        #     done_flag = done
        #     if episode_timesteps == env._max_episode_steps:
        #         done_flag = False

        #     replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done_flag)
        #     state = next_state

        #     if done:
        #         state = env.reset()
        #         episode_timesteps = 0
        #         episode_return = 0

        # # pretraining the extractor, extractor's trainable variables are incomplete until the extractor has been trained once.
        # if args.gin is not None:
        #     print("Pretrain: I am pretraining the extractor!")
        #     for i in range(2):

        #         tf.summary.experimental.set_step(i - args.pre_train_step)
        #         sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones = replay_buffer.sample(
        #             batch_size=batch_size)
        #         sample_next_actions = [env.action_space.sample() for k in range(batch_size)]
                
        #         pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred = extractor.train(extractor_target, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)

        #     print("OFENet Projection's network structure:")
        #     tvars = extractor.projection.trainable_variables
        #     for var in tvars:
        #         print(" name = %s, shape = %s" % (var.name, var.shape))
    
        # test model in environme nts with different masses and frictions
        dir_log = make_output_dir(dir_root=dir_root, exp_name=exp_name, env_name=env_name, 
                                            seed=seed, ignore_errors=args.force)
        model_save_dir = os.path.join(dir_log, 'model')
        algo = ''  
        if 'ofePaper' in model_save_dir:
            algo = 'ofePaper'
            # Load extractor
            extractor_ofePaper = feature_extractor_ofePaper(env_name, dim_state, dim_action)
            extractor_ofePaper.load_weights(os.path.join(model_save_dir,'extractor_model'))
            logger.info("Feature extractor's model has been loaded.")
            # Load policy
            policy = make_policy(policy=policy_name, 
                                    env_name=env_name, 
                                    extractor=extractor_ofePaper, 
                                    units=args.sac_units)
            policy.load(model_save_dir)
            logger.info("Policy's model has been loaded.")
        
        elif 'raw' in model_save_dir:
            algo = 'raw'
             # Load extractor
            extractor = DummyFeatureExtractor(dim_state=dim_state, dim_action=dim_action)
            logger.info("Feature extractor's model has been loaded.")

            # Load policy
            policy = make_policy(policy=policy_name, 
                                env_name=env_name, 
                                extractor=extractor, 
                                units=args.sac_units)
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
                                units=args.sac_units)
            policy.load(model_save_dir)
            logger.info("Policy's model has been loaded.")

        img_save_dir = './test/img_ablation/periodicity'
        img_save_sub_dir = policy_name + '-' + algo + '-' + dir_of_env[env_name]

        
        # test model
        try:
            exp_data = pd.read_csv(os.path.join(img_save_dir, img_save_sub_dir, '{}_state_action_data.csv'.format(dir_of_env[env_name])))  # pd.read_csv读取以逗号为分割符的文件
        except:
            print('Could not read from %s'%os.path.join(img_save_dir, img_save_sub_dir, '{}_state_action_data.csv'.format(dir_of_env[env_name])))
            continue

        exp_data_len = len(exp_data)
        horizon = 80
        s_future_data = tf.cast(np.array(exp_data.iloc[exp_data_len - horizon + 1: exp_data_len, 1: dim_state + 1]), dtype=tf.float32)
        # input = exp_data.iloc[exp_data_len - horizon].values.tolist()
        input_s = np.array(exp_data.iloc[[exp_data_len - horizon], 1: dim_state + 1])
        input_a = np.array(exp_data.iloc[[exp_data_len - horizon], dim_state + 1: dim_state + dim_action + 1])

        with tf.device("/gpu:{}".format(0)):
            fourier_pred_re, fourier_pred_im = extractor([input_s, input_a])
            fourier_pred_re = np.squeeze(fourier_pred_re)
            fourier_pred_im = np.squeeze(fourier_pred_im)

            con_gamma = tf.cast(np.diag([pow(args.discount, k) for k in range(horizon - 1)]), dtype=tf.float32)
            # end = int(args.dim_discretize*0.5 + 1)  # predict fourier function in [0,\pi]
            ratio = 2 * np.pi / args.dim_discretize
            con = tf.matmul(tf.cast(ratio * np.expand_dims(np.arange(args.dim_discretize), axis=1), dtype=tf.float32), \
                            tf.cast(np.expand_dims(np.arange(horizon - 1), axis=0), dtype=tf.float32))

            con_re = tf.math.cos(con)
            con_im = - tf.math.sin(con)
            fourier_true_re = tf.matmul(tf.matmul(con_re, con_gamma), s_future_data)
            fourier_true_im = tf.matmul(tf.matmul(con_im, con_gamma), s_future_data)

            z = tf.complex(fourier_pred_re, fourier_pred_im)
            fourier_pred_mod = np.abs(z) / (horizon / 2)
            fourier_pred_angle = np.angle(z)

            z = tf.complex(fourier_true_re, fourier_true_im)
            fourier_true_mod = np.abs(z) / horizon
            for i in range(dim_state):
                if env_name == 'Humanoid-v2' and np.amax(fourier_true_mod[:, i]) > 0.005:
                    fourier_true_mod[:, i] = fourier_true_mod[:, i] / np.amax(fourier_true_mod[:, i]) * 0.5
                elif np.amax(fourier_true_mod[:, i]) > 0.1:
                    fourier_true_mod[:, i] = fourier_true_mod[:, i] / np.amax(fourier_true_mod[:, i]) * 0.1
            fourier_true_angle = np.angle(z)

            # fft_s_future = fft(s_future_data)
            # abs_fs = np.abs(fft_s_future)
            # angle_fs = np.angle(fft_s_future)
            # normalization_fs = abs_fs / horizon

            # fourier_error_mod = fourier_true_mod[:,:get_target_dim(env_name)] - fourier_pred_mod
            # fourier_error_angle = fourier_true_angle[:,:get_target_dim(env_name)] - fourier_pred_angle

            # fourier_pred_re_proj = extractor.projection(fourier_pred_re)
            # fourier_pred_im_proj = extractor.projection(fourier_pred_im)
            # fourier_true_re_proj = extractor.projection(fourier_true_re)
            # fourier_true_im_proj = extractor.projection(fourier_true_im)

            # loss_fun = tf.keras.losses.CosineSimilarity(axis=-1)
            # fourier_csError_re = loss_fun(fourier_pred_re_proj, fourier_true_re_proj)
            # fourier_csError_im = loss_fun(fourier_pred_im_proj, fourier_true_im_proj)

        # types = ['true', 'error']
        types = ['true']
        complex_types = ['module']
        # complex_types = ['module', 'angle']
        for complex_type in complex_types:
            
            for type0 in types:
                
                if not os.path.exists(os.path.join(img_save_dir, img_save_sub_dir, '{}_fourier_{}'.format(type0, complex_type))):
                    os.mkdir(os.path.join(img_save_dir, img_save_sub_dir, '{}_fourier_{}'.format(type0, complex_type)))

                for i in range(dim_state):
                    fig, ax = plt.subplots()
                    logger.info('Drawing {}\'s state{}\'s {} fourier {}\n'.format(env_name.split("-")[0], i, type0, complex_type))

                    if complex_type == 'module':
                        if type0 == 'true':
                            ax.plot(ratio * np.arange(args.dim_discretize), fourier_true_mod[:, i])
                            # ax.plot(ratio * np.arange(end), fourier_pred_mod[:, i])
                            # ax.legend(['true', 'pred'])
                            # plt.ylim((0, 0.1))
                        elif type0 == 'error':
                            ax.plot(ratio * np.arange(args.dim_discretize), fourier_error_mod[:, i])
                            ax.legend(['error'])

                    elif complex_type == 'angle':
                        if type0 == 'true':
                            ax.plot(ratio * np.arange(args.dim_discretize), fourier_true_angle[:, i])
                            # ax.plot(ratio * np.arange(end), fourier_pred_angle[:, i])
                            # ax.legend(['true', 'pred'])
                        elif type0 == 'error':
                            ax.plot(ratio * np.arange(args.dim_discretize), fourier_error_angle[:, i])
                            ax.legend(['error'])
                    
                    ax.set_xlabel('Omega')
                    ax.set_ylabel('state{}\'s fourier {}'.format(i, complex_type))
                    ax.set_title('{}\'s state{} FT {} of the last {} states in an episode'.format(env_name.split("-")[0], i, complex_type, horizon))

                    plt.savefig(os.path.join(img_save_dir, img_save_sub_dir, '{}_fourier_{}'.format(type0, complex_type), '{}_state{}\'s_{}_fourier_{}'.format(dir_of_env[env_name], i, type0, complex_type)), dpi=300)
                    plt.close ('all')

        # # drawing states
        # for i in range(dim_state):
        #     fig, ax = plt.subplots()
        #     logger.info('Drawing {}\'s state{}\n'.format(env_name.split("-")[0], i))

        #     ax.plot(np.arange(exp_data_len - horizon + 1, exp_data_len), s_future_data[:, i])
        #     ax.set_xlabel('InterStep')
        #     ax.set_ylabel('state{}'.format(i))
        #     ax.set_title('{}\'s state{} in an episode'.format(env_name.split("-")[0], i))
        #     ax.legend(['state{}'.format(i)])

        #     plt.savefig(os.path.join(img_save_dir, img_save_sub_dir, 'states', '{}_state{}'.format(dir_of_env[env_name], i)), dpi=300)
            plt.close ('all')
            
        logger.info('Done')


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )
    main()