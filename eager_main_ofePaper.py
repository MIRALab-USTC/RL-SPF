import argparse
import logging
import os
import shutil
import sys
import time

import gin
import gym
import numpy as np
import tensorflow as tf
import datetime, pytz

import src.util.gin_utils as gin_utils
from src.aux.dummy_extractor import DummyFeatureExtractor
from src.aux.munk_extractor import MunkNet
from src.aux.network_ofePaper import OFENet
from src.policy import DDPG
from src.policy import PPO
from src.policy import SAC
from src.policy import TD3
from src.util import misc
from src.util import replay
from src.util.misc import get_target_dim, make_ofe_name, get_default_steps

misc.set_gpu_device_growth()
dir_of_env = {'HalfCheetah-v2': 'hc', 
            'Hopper-v2': 'hopper',  
            'Walker2d-v2': 'walker', 
            'Ant-v2': 'ant', 
            'Swimmer-v2': 'swim', 
            'Humanoid-v2': 'human',
            "InvertedDoublePendulum-v2": 'IDPen'}


def evaluate_policy(env, policy, eval_episodes=10):
    avg_reward = 0.
    episode_length = []

    for _ in range(eval_episodes):
        state = env.reset()
        cur_length = 0

        done = False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = env.step(action)
            avg_reward += reward
            cur_length += 1

        episode_length.append(cur_length)

    avg_reward /= eval_episodes
    avg_length = np.average(episode_length)
    return avg_reward, avg_length


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
        policy = SAC.SAC(state_dim, action_dim, max_action, feature_extractor=extractor, scale_reward=scale_reward,
                         actor_units=n_units, q_units=n_units, v_units=n_units)
    elif policy == "DDPG":
        policy = DDPG.DDPG(state_dim, action_dim, max_action, feature_extractor=extractor)
        print("We use DDPG algorithm!")
    elif policy == "PPO":
        policy = PPO.PPO(state_dim, action_dim, max_action, feature_extractor=extractor)
        print("We use PPO algorithm!")
    elif policy == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(400, 300), feature_extractor=extractor)
    elif policy == "TD3small":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(256, 256), feature_extractor=extractor)
    else:
        raise ValueError("invalid policy {}".format(policy))

    return policy


def make_output_dir(dir_root, exp_name, env_name, seed, ignore_errors):
    seed_name = "seed{}".format(seed)

    dir_log = os.path.join(dir_root, "log_{}".format(dir_of_env[env_name]), exp_name, seed_name)
    
    for cur_dir in [dir_log]:
        if os.path.exists(cur_dir):
            if ignore_errors:
                shutil.rmtree(cur_dir, ignore_errors=True)
            else:
                raise ValueError("output directory {} exists".format(cur_dir))

        os.makedirs(cur_dir)

    return dir_log


@gin.configurable
def feature_extractor(env_name, dim_state, dim_action, normalizer='batch', name=None, skip_action_branch=False):
    logger = logging.getLogger(name="main")
    logger.info("Use Extractor {}".format(name))

    if name == "OFE":
        target_dim = get_target_dim(env_name)
        extractor = OFENet(dim_state=dim_state, dim_action=dim_action,
                           dim_output=target_dim, normalizer=normalizer, skip_action_branch=skip_action_branch)
        # print network parameters of OFENet
        print("OFENet's network structure:\n")
        tvars = extractor.trainable_variables
        for var in tvars:
            print(" name = %s, shape = %s" % (var.name, var.shape))
    elif name == "Munk":
        extractor = MunkNet(dim_state=dim_state, dim_action=dim_action)
    else:
        extractor = DummyFeatureExtractor(dim_state=dim_state, dim_action=dim_action)

    return extractor


def main(args):
    logger = logging.Logger(name="main")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s',
                                           datefmt="%m/%d %I:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # CONSTANTS
    if args.gin is not None:
        gin.parse_config_file(args.gin)

    max_steps = args.steps
    summary_freq = 1000
    eval_freq = 5000
    random_collect = 10000

    if eval_freq % summary_freq != 0:
        logger.error("eval_freq must be divisible by summary_freq.")
        sys.exit(-1)

    max_steps = args.steps = get_default_steps(args.env)

    env_name = args.env
    policy_name = args.policy
    batch_size = args.batch_size
    seed = args.seed
    dir_root = args.dir_root

    exp_name = make_exp_name(args)
    logger.info("Start Experiment {}".format(exp_name))

    dir_log = make_output_dir(dir_root=dir_root, exp_name=exp_name, env_name=args.env, seed=seed,
                                             ignore_errors=args.force)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    # Set seeds
    env.seed(seed)
    eval_env.seed(seed + 1000)
    # tf.set_random_seed(args.seed)
    np.random.seed(seed)

    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    extractor = feature_extractor(env_name, dim_state, dim_action, args.normalizer)

    # Makes a summary writer before graph construction
    writer = tf.summary.create_file_writer(dir_log)
    writer.set_as_default()

    policy = make_policy(policy=policy_name, env_name=env_name, extractor=extractor, units=args.sac_units)

    replay_buffer = replay.ReplayBuffer(state_dim=dim_state, action_dim=dim_action, capacity=1000000)

    gin_utils.write_gin_to_summary(dir_log, global_step=0)

    total_timesteps = np.array(0, dtype=np.int32)
    episode_timesteps = 0
    episode_return = 0
    state = env.reset()

    logger.info("collecting random {} transitions".format(random_collect))

    print("Initialization: I am collecting samples randomly!")
    for i in range(random_collect):
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

    # pretraining the extractor

    print("Pretrain: I am pretraining the extractor!")
    for i in range(random_collect):
        sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones = replay_buffer.sample(
            batch_size=batch_size)
        extractor.train(sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones)

    state = np.array(state, dtype=np.float32)
    prev_calc_time = time.time()
    prev_calc_step = random_collect

    print("Train: I am starting to train myself!")
    should_summary = lambda: tf.equal(total_timesteps % summary_freq, 0)
    with tf.summary.record_if(should_summary):
        for cur_steps in range(random_collect + 1, max_steps + 1):
            action = policy.select_action_noise(state)
            action = action.clip(env.action_space.low, env.action_space.high)

            next_state, reward, done, _ = env.step(action)
            episode_timesteps += 1
            episode_return += reward
            total_timesteps += 1
            tf.summary.experimental.set_step(total_timesteps)

            done_flag = done

            # done is valid, when an episode is not finished by max_step.
            if episode_timesteps == env._max_episode_steps:
                done_flag = False

            replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done_flag)
            state = next_state

            if done:
                state = env.reset()

                logger.info("Time {} : Sample Steps {} Reward {}".format(int(total_timesteps), episode_timesteps,
                                                                         episode_return))

                with tf.summary.record_if(True):
                    tf.summary.scalar(name="performance/exploration_steps", data=episode_timesteps,
                                      description="Exploration Episode Length")
                    tf.summary.scalar(name="performance/exploration_return", data=episode_return,
                                      description="Exploration Episode Return")

                episode_timesteps = 0
                episode_return = 0

            sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones = replay_buffer.sample(
                batch_size=batch_size)
            extractor.train(sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones)

            policy.train(replay_buffer, batch_size=batch_size)

            if cur_steps % eval_freq == 0:
                duration = time.time() - prev_calc_time
                duration_steps = cur_steps - prev_calc_step
                throughput = duration_steps / float(duration)

                logger.info("Throughput {:.2f}   ({:.2f} secs)".format(throughput, duration))

                cur_evaluate, average_length = evaluate_policy(eval_env, policy)
                logger.info("Evaluate Time {} : Average Reward {}".format(int(total_timesteps), cur_evaluate))
                tf.summary.scalar(name="performance/evaluate_return", data=cur_evaluate,
                                  description="Evaluate for test dataset")
                tf.summary.scalar(name="performance/evaluate_steps", data=average_length,
                                  description="Step length during evaluation")
                tf.summary.scalar(name="throughput", data=throughput, description="Throughput. Steps per Second.")

                prev_calc_time = time.time()
                prev_calc_step = cur_steps

            # store model
            if args.save_model == True and cur_steps % args.save_freq == 0:
                model_save_dir = os.path.join(dir_log, 'model')
                policy.save(model_save_dir)

                if args.gin is not None:
                    extractor.save_weights(os.path.join(model_save_dir,'extractor_model'))
                    print('Models have been saved.')

        # store model
        tf.summary.flush()
        # checkpoint_manager.save(checkpoint_number=tf.constant(cur_steps, dtype=tf.int64))


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--steps", default=1000000, type=int)
    parser.add_argument("--sac-units", default=256, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--gin", default=None)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--force", default=False, action="store_true",
                        help="remove existed directory")
    parser.add_argument("--dir-root", default="output", type=str)
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--save_freq", default=100000, type=int)
    parser.add_argument("--normalizer", default='layer', type=str, choices=['layer', 'batch'])
    args = parser.parse_args()
    
    main(args)

