# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import argparse
import logging
import sys
import time

import gin
import gym
import numpy as np
import tensorflow as tf
from cpprb import ReplayBuffer
from scipy.signal import lfilter

import src.util.gin_utils as gin_utils
from src.aux.dummy_extractor import DummyFeatureExtractor
from src.policy.PPO import PPO
from src.tool.eager_main import make_exp_name, make_output_dir, \
    feature_extractor
from src.util import replay
from src.util.misc import set_gpu_device_growth, EmpiricalNormalization

set_gpu_device_growth()


class Trainer:
    def __init__(self, policy, env, eval_env, checkpoint_manager,
                 dim_state, dim_action, ofe_batchsize=256):
        self._policy = policy
        self._ofe_batchsize = ofe_batchsize
        self._policy = policy
        self._normalizer = EmpiricalNormalization(shape=(dim_state,), clip_threshold=5.)
        self._env = env
        self._eval_env = eval_env
        self._prepare_buffer(dim_state, dim_action)
        self._checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(__name__)

    def _prepare_buffer(self, dim_state, dim_action, ofe_buffer_size=int(1e6)):
        replay_buffer_config = {
            "size": self._policy.horizon,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": (dim_state,)},
                "next_obs": {"shape": (dim_state,)},
                "act": {"shape": (dim_action,)},
                "logp": {},
                "ret": {},
                "adv": {},
                "done": {}}}
        self.replay_buffer = ReplayBuffer(**replay_buffer_config)

        local_buffer_kwargs = {
            "size": self._env._max_episode_steps,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": (dim_state,)},
                "next_obs": {"shape": (dim_state,)},
                "act": {"shape": (dim_action,)},
                "rew": {},
                "logp": {},
                "val": {},
                "done": {}}}
        self.local_buffer = ReplayBuffer(**local_buffer_kwargs)

        ofe_buffer_kwargs = {
            "size": ofe_buffer_size,
            "default_dtype": np.float32,
            "env_dict": {
                "obs": {"shape": (dim_state,)},
                "next_obs": {"shape": (dim_state,)},
                "act": {"shape": (dim_action,)}}}
        self.ofe_buffer = ReplayBuffer(**ofe_buffer_kwargs)

    def finish_horizon(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        if self.local_buffer.get_stored_size() == 0:
            return None

        samples = self.local_buffer._encode_sample(
            np.arange(self.local_buffer.get_stored_size()))
        rews = np.append(samples["rew"], last_val)
        vals = np.append(samples["val"], last_val)

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + self._policy.discount * vals[1:] - vals[:-1]
        advs = discount_cumsum(
            deltas, self._policy.discount * self._policy.lam)

        # Rewards-to-go, to be targets for the value function
        rets = discount_cumsum(rews, self._policy.discount)[:-1]
        self.replay_buffer.add(
            obs=samples["obs"], next_obs=samples["next_obs"],
            act=samples["act"], done=samples["done"],
            ret=rets, adv=advs, logp=np.squeeze(samples["logp"]))
        self.local_buffer.clear()

    def run(self, extractor, random_collect=10000, max_steps=2000000, eval_freq=5000):
        logger = logging.getLogger(name="main")

        total_steps = np.array(0, dtype=np.int32)
        total_steps += random_collect
        n_episode = 0
        prev_calc_time = time.time()
        prev_calc_step = random_collect
        episode_timesteps = 0
        episode_return = 0

        obs = self._env.reset()
        tf.summary.experimental.set_step(total_steps)

        while total_steps < max_steps:
            # Collect samples
            for _ in range(self._policy.horizon):
                total_steps += 1
                episode_timesteps += 1
                tf.summary.experimental.set_step(total_steps)

                input_obs = self._normalizer(np.expand_dims(obs, axis=0), update=False)[0]
                act, logp, val = self._policy.get_action_and_val(input_obs)
                next_obs, reward, done, _ = self._env.step(act)
                episode_return += reward

                done_flag = done
                if episode_timesteps == self._env._max_episode_steps:
                    done_flag = False
                self.local_buffer.add(
                    obs=obs, act=act, next_obs=next_obs,
                    rew=reward, done=done_flag, logp=logp, val=val)
                self.ofe_buffer.add(obs=obs, act=act, next_obs=next_obs)
                obs = next_obs

                if done or episode_timesteps == self._env._max_episode_steps:
                    self.finish_horizon()
                    obs = self._env.reset()
                    n_episode += 1
                    logging.info(
                        "Total Epi: {0: 5} Steps: {1: 7} Episode Steps: {2: 5} Return: {3: 5.4f}".format(
                            n_episode, int(total_steps), episode_timesteps, episode_return))
                    tf.summary.scalar(name="loss/exploration_steps", data=episode_timesteps,
                                      description="Exploration Episode Length")
                    tf.summary.scalar(name="loss/exploration_return", data=episode_return,
                                      description="Exploration Episode Return")
                    episode_timesteps = 0
                    episode_return = 0

                if total_steps % eval_freq == 0:
                    duration = time.time() - prev_calc_time
                    duration_steps = total_steps - prev_calc_step
                    throughput = duration_steps / float(duration)

                    logger.info("Throughput {:.2f}   ({:.2f} secs)".format(throughput, duration))

                    cur_evaluate, average_length = self.evaluate_policy()
                    logger.info("Evaluate Time {} : Average Reward {}".format(int(total_steps), cur_evaluate))
                    tf.summary.scalar(name="loss/evaluate_return", data=cur_evaluate,
                                      description="Evaluate for test dataset")
                    tf.summary.scalar(name="loss/evaluate_steps", data=average_length,
                                      description="Step length during evaluation")
                    tf.summary.scalar(name="throughput", data=throughput,
                                      description="Throughput. Steps per Second.")

                    prev_calc_time = time.time()
                    prev_calc_step = total_steps.copy()
            # TODO: not next_val?
            self.finish_horizon(last_val=val)
            # Finished collecting samples

            # Train actor critic
            samples = self.replay_buffer._encode_sample(np.arange(self._policy.horizon))
            mean_adv = np.mean(samples["adv"])
            std_adv = np.std(samples["adv"])

            # Update normalizer
            self._normalizer.experience(samples["obs"])

            with tf.summary.record_if(total_steps % (self._policy.horizon * 10) == 0):
                # train OFE
                for _ in range(self._policy.horizon):
                    samples = self.ofe_buffer.sample(self._ofe_batchsize)
                    samples["obs"] = self._normalizer(samples["obs"], update=False)
                    samples["next_obs"] = self._normalizer(samples["next_obs"], update=False)
                    extractor.train(
                        states=samples["obs"],
                        actions=samples["act"],
                        next_states=samples["next_obs"],
                        rewards=None,
                        dones=None)
                # train policy
                for _ in range(self._policy.n_epoch):
                    samples = self.replay_buffer._encode_sample(
                        np.random.permutation(self._policy.horizon))
                    samples["obs"] = self._normalizer(samples["obs"], update=False)
                    adv = (samples["adv"] - mean_adv) / (std_adv + 1e-8)
                    for idx in range(int(self._policy.horizon / self._policy.batch_size)):
                        target = slice(idx * self._policy.batch_size,
                                       (idx + 1) * self._policy.batch_size)
                        self._policy.train(
                            raw_states=samples["obs"][target],
                            actions=samples["act"][target],
                            advantages=adv[target],
                            logp_olds=samples["logp"][target],
                            returns=samples["ret"][target])

        # store model
        tf.summary.flush()
        self._checkpoint_manager.save(checkpoint_number=tf.constant(max_steps, dtype=tf.int64))

    def evaluate_policy(self, eval_episodes=10):
        avg_reward = 0.
        episode_length = []

        for _ in range(eval_episodes):
            state = self._eval_env.reset()
            state = self._normalizer(np.expand_dims(state, axis=0), update=False)[0]
            cur_length = 0

            done = False
            while not done:
                action = self._policy.select_action(np.array(state))
                state, reward, done, _ = self._eval_env.step(action)
                state = self._normalizer(np.expand_dims(state, axis=0), update=False)[0]
                avg_reward += reward
                cur_length += 1

            episode_length.append(cur_length)

        avg_reward /= eval_episodes
        avg_length = np.average(episode_length)
        return avg_reward, avg_length

    def pretrain_ofe(self, extractor, dim_state, dim_action, random_collect=10000):
        replay_buffer = replay.ReplayBuffer(
            state_dim=dim_state, action_dim=dim_action, capacity=random_collect)
        episode_timesteps = 0

        state = self._env.reset()
        for i in range(random_collect):
            action = self._env.action_space.sample()
            next_state, reward, done, _ = self._env.step(action)

            episode_timesteps += 1

            done_flag = done
            if episode_timesteps == self._env._max_episode_steps:
                done_flag = False

            replay_buffer.add(state=state, action=action, next_state=next_state, reward=reward, done=done_flag)
            state = next_state

            if done:
                state = self._env.reset()
                episode_timesteps = 0

        for i in range(random_collect):
            sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones = replay_buffer.sample(
                batch_size=64)
            extractor.train(sample_states, sample_actions, sample_next_states, sample_rewards, sample_dones)


def discount_cumsum(x, discount):
    """
    Forked from rllab for computing discounted cumulative sums of vectors.

    :param x (np.ndarray or tf.Tensor)
        vector of [x0, x1, x2]
    :return output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return lfilter(
        b=[1],
        a=[1, float(-discount)],
        x=x[::-1],
        axis=0)[::-1]


def main():
    logger = logging.Logger(name="main")
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s',
                                           datefmt="%m/%d %I:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="PPO")
    parser.add_argument("--env", default="HalfCheetah-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--steps", default=2000000, type=int)
    parser.add_argument("--gin", default=None)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--force", default=False, action="store_true", help="remove existed directory")
    parser.set_defaults(force=True)
    parser.set_defaults(seed=0)
    parser.add_argument("--dir-root", default="output", type=str)
    args = parser.parse_args()

    # CONSTANTS
    if args.gin is not None:
        gin.parse_config_file(args.gin)

    summary_freq = 1000
    eval_freq = 5000
    random_collect = 10000

    if eval_freq % summary_freq != 0:
        logger.error("eval_freq must be divisible by summary_freq.")
        sys.exit(-1)

    env_name = args.env
    seed = args.seed
    dir_root = args.dir_root

    exp_name = make_exp_name(args)
    logger.info("Start Experiment {}".format(exp_name))

    dir_log, dir_parameter = make_output_dir(dir_root=dir_root, exp_name=exp_name, seed=seed,
                                             ignore_errors=args.force)

    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    # Set seeds
    env.seed(seed)
    eval_env.seed(seed + 1000)
    np.random.seed(seed)

    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    extractor = feature_extractor(env_name, dim_state, dim_action, skip_action_branch=True)

    if isinstance(extractor, DummyFeatureExtractor):
        random_collect = 0

    # Makes a summary writer before graph construction
    # https://github.com/tensorflow/tensorflow/issues/26409
    writer = tf.summary.create_file_writer(dir_log)
    writer.set_as_default()

    policy = PPO(
        state_dim=dim_state,
        action_dim=dim_action,
        max_action=env.action_space.high[0],
        feature_extractor=extractor,
        horizon=2048,
        batch_size=64,
        actor_units=(64, 64),
        critic_units=(64, 64),
        n_epoch=10,
        lr=3e-4,
        discount=0.995,
        lam=0.97,
        gpu=0)
    gin_utils.write_gin_to_summary(dir_log, global_step=0)

    checkpoint = tf.train.Checkpoint(policy=policy)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    directory=dir_parameter,
                                                    max_to_keep=1)

    trainer = Trainer(policy, env, eval_env, checkpoint_manager,
                      dim_state, dim_action)
    if random_collect > 0:
        trainer.pretrain_ofe(extractor, dim_state, dim_action, random_collect)
    trainer.run(extractor, random_collect=random_collect, max_steps=args.steps)


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s')

    main()
