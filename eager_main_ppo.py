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
import datetime, pytz

import src.util.gin_utils as gin_utils
from src.aux.dummy_extractor import DummyFeatureExtractor
from src.aux.munk_extractor import MunkNet
from src.aux.network import OFENet
from src.policy import DDPG
from src.policy import PPO
from src.policy import SAC
from src.policy import TD3
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
        else:
            raise ValueError("invalid extractor name {}".format(extractor_name))
    else:
        ofe_name = "raw"

    env_name = args.env.split("-")[0]
    exp_name = "{}_{}_{}".format(env_name, args.policy, ofe_name)

    if args.gin is not None:
        exp_name = exp_name + "_tau" + str(args.tau) + "_freq" + str(args.target_update_freq) + "_up" + str(args.update_every)

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
        print("We use DDPG algorithm!")
    elif policy == "PPO":
        policy = PPO.PPO(state_dim, action_dim, max_action, feature_extractor=extractor)
        print("We use PPO algorithm!")
    elif policy == "TD3":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(400, 300), feature_extractor=extractor)
        print("We use TD3 algorithm!")
    elif policy == "TD3small":
        policy = TD3.TD3(state_dim, action_dim, max_action, layer_units=(256, 256), feature_extractor=extractor)
        print("We use TDsmall algorithm!")
    else:
        raise ValueError("invalid policy {}".format(policy))

    return policy

def make_output_dir(dir_root, exp_name, env_name, seed, ignore_errors):
    seed_name = "seed{}".format(seed)

    dir_log = os.path.join(dir_root, "log_{}".format(dir_of_env[env_name]), exp_name, seed_name)

    for cur_dir in [dir_log]:  # If the same file name exists, delete the directory recursively
        if os.path.exists(cur_dir):
            if ignore_errors:
                shutil.rmtree(cur_dir, ignore_errors=True)  # delete the directory recursively
            else:
                raise ValueError("output directory {} exists".format(cur_dir))

        os.makedirs(cur_dir)

    return dir_log


@gin.configurable
def feature_extractor(extractor_kwargs=dict(), name=None, skip_action_branch=False):
    logger = logging.getLogger(name="main")
    logger.info("Use Extractor {}".format(name))

    if name == "OFE":
        extractor = OFENet(**extractor_kwargs, skip_action_branch=skip_action_branch)
        extractor_target = OFENet(**extractor_kwargs, skip_action_branch=skip_action_branch)
        target_update.update_target_variables(extractor_target.weights, extractor.weights)
        # print network parameters of OFENet
        print("OFENet's network structure:\n")
        tvars = extractor.trainable_variables
        for var in tvars:
            print(" name = %s, shape = %s" % (var.name, var.shape))
    else:
        extractor = DummyFeatureExtractor(dim_state=dim_state, dim_action=dim_action)

    return extractor, extractor_target


def main(args):
    logger = logging.Logger(name="main")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s',
                                           datefmt="%m/%d %I:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    args_text = '\n'.join([f'{k:<20}: {v}' for k, v in vars(args).items()])
    logger.info(args_text)

    start_time = time.time()

    # CONSTANTS
    if args.gin is not None:
        gin.parse_config_file(args.gin)

    # make sure the normal execution of replay_buffer.sample_with_Hth_states
    assert args.random_collect - args.dim_discretize > args.batch_size

    if args.eval_freq % args.summary_freq != 0:
        logger.error("eval_freq must be divisible by summary_freq.")
        sys.exit(-1)

    seed = args.seed

    max_steps = args.steps = get_default_steps(args.env)
    epochs = max_steps // args.steps_per_epoch + 1

    total_steps = args.steps_per_epoch*epochs
    steps_per_epoch = args.steps_per_epoch
    save_freq = args.save_freq


    env_name = args.env
    policy_name = args.policy
    batch_size = args.batch_size
    dir_root = args.dir_root

    exp_name = make_exp_name(args)
    logger.info("Start Experiment {}".format(exp_name))

    dir_log = make_output_dir(dir_root=dir_root, exp_name=exp_name, env_name=args.env, 
                                seed=seed, ignore_errors=args.force)
 
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 1000)

    # Set seeds
    env = gym.make(env_name)
    env.seed(seed)
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
        "normalizer": args.normalizer
    }
    extractor, extractor_target = feature_extractor(extractor_kwargs=extractor_kwargs)


    # Makes a summary writer before graph construction
    writer = tf.summary.create_file_writer(dir_log)
    writer.set_as_default()

    tf.summary.text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        step = 0
    )

    # Initialize policy
    policy = make_policy(policy=policy_name, env_name=env_name, extractor=extractor, units=args.sac_units)

    replay_buffer = replay.PPOBuffer(obs_dim=dim_state, act_dim=dim_action, size=steps_per_epoch, gamma=args.discount, lam=args.lam)

    gin_utils.write_gin_to_summary(dir_log, global_step=0)

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

        replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=0, logp=0)
        state = next_state

        if done:
            state = env.reset()
            episode_timesteps = 0
            episode_return = 0

    # pretraining the extractor
    if args.gin is not None:
        print("Pretrain: I am pretraining the extractor!")
        for i in range(args.pre_train_step):

            tf.summary.experimental.set_step(i - args.pre_train_step)
            sample_states, sample_actions, sample_next_states, _, sample_dones = replay_buffer.sample(
                batch_size=batch_size)
            sample_next_actions, _ = policy.get_action(sample_next_states)

            pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred = extractor.train(extractor_target, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)

        print("OFENet Projection's network structure:")
        tvars = extractor.projection.trainable_variables
        for var in tvars:
            print(" name = %s, shape = %s" % (var.name, var.shape))
            
    state = np.array(state, dtype=np.float32)
    prev_calc_time = time.time()
    prev_calc_step = args.pre_train_step
    replay_buffer.get()

    print("Train: I am starting to train myself!")
    # should_summary = lambda: tf.equal(total_timesteps % args.summary_freq, 0)
    # with tf.summary.record_if(should_summary):
    for cur_steps in range(total_steps):
        action, logp, v_t = policy.get_action_and_val(state)

        # Step the env
        next_state, reward, done, _ = env.step(action)
        episode_timesteps += 1
        episode_return += reward
        total_timesteps += 1
        tf.summary.experimental.set_step(total_timesteps)

        done_flag = done

        # done is valid, when an episode is not finished by max_step.
        if episode_timesteps == env._max_episode_steps:
            done_flag = False

        replay_buffer.add(obs=state, act=action, obs2=next_state, rew=reward, done=done_flag, val=v_t, logp=logp)
        state = next_state

        if done or (episode_timesteps == env._max_episode_steps) or (cur_steps + 1) % steps_per_epoch == 0:
            # if trajectory didn't reach terminal state, bootstrap value target
            last_val = 0 if done else policy.get_val(state)
            replay_buffer.finish_path(last_val)
            state = env.reset()

            logger.info("Time {} : Sample Steps {} Reward {}".format(int(total_timesteps), episode_timesteps,
                                                                        episode_return))

            with tf.summary.record_if(True):
                tf.summary.scalar(name="performance/exploration_steps", data=episode_timesteps,
                                    description="Exploration Episode Length")
                tf.summary.scalar(name="performance/exploration_return", data=episode_return,
                                    description="Exploration Episode Return")

                if args.gin is not None:
                    tf.summary.scalar(name="loss/predictor_Loss", data=pred_loss)
                    tf.summary.scalar(name="loss/predictor_Re_Loss", data=pred_re_loss)
                    tf.summary.scalar(name="loss/predictor_Im_Loss", data=pred_im_loss)
                    # tf.summary.scalar(name="loss/predictor_Inv_Loss", data=inv_loss)

            episode_timesteps = 0
            episode_return = 0

        update_every = args.update_every
        if args.gin is not None and cur_steps % update_every == 0:
            for _ in range(update_every):
                sample_states, sample_actions, sample_next_states, _, sample_dones = replay_buffer.sample(
                    batch_size=batch_size)
                sample_next_actions, _ = policy.get_action(sample_next_states)
                pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred = extractor.train(extractor_target, sample_states, sample_actions, sample_next_states, sample_next_actions, sample_dones)
        if args.gin is not None and cur_steps % args.target_update_freq == 0:
            target_update.update_target_variables(extractor_target.weights, extractor.weights, tau=args.tau)

        if (cur_steps + 1) % steps_per_epoch == 0:
            policy.train(replay_buffer)

        if cur_steps % args.eval_freq == 0:
            duration = time.time() - prev_calc_time
            duration_steps = cur_steps - prev_calc_step
            throughput = duration_steps / float(duration)

            logger.info("Throughput {:.2f}   ({:.2f} secs)".format(throughput, duration))

            cur_evaluate, average_length = evaluate_policy(eval_env, policy)
            logger.info("Evaluate Time {} : Average Reward {}".format(int(total_timesteps), cur_evaluate))
            with tf.summary.record_if(True):
                tf.summary.scalar(name="performance/evaluate_return", data=cur_evaluate,
                                description="Evaluate for test dataset")
                tf.summary.scalar(name="performance/evaluate_steps", data=average_length,
                                description="Step length during evaluation")
                tf.summary.scalar(name="throughput", data=throughput, description="Throughput. Steps per Second.")

                if args.gin is not None:
                    logger.info("Evaluate Time {} : recording predictor loss".format(int(total_timesteps)))
                    tf.summary.scalar(name="loss/predictor_Loss", data=pred_loss)
                    tf.summary.scalar(name="loss/predictor_Re_Loss", data=pred_re_loss)
                    tf.summary.scalar(name="loss/predictor_Im_Loss", data=pred_im_loss)
                   
            prev_calc_time = time.time()
            prev_calc_step = cur_steps

        # store model
        if args.save_model == True and cur_steps % args.save_freq == 0:
            model_save_dir = os.path.join(dir_log, 'model')
            policy.save(model_save_dir)
            # replay_buffer.save(dir_log)
            # print('Reply buffer have been saved.')

            if args.gin is not None:
                extractor.save_weights(os.path.join(model_save_dir,'extractor_model'))
                extractor_target.save_weights(os.path.join(model_save_dir,'extractor_target_model'))
                print('Models have been saved.')

        tf.summary.flush()


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s'
                        )

    args = parse_args()
    main(args)
