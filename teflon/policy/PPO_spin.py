# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

# Forked from https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/ppo.py

import os
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from teflon.util.mpi_tf2 import MpiAdamOptimizer, sync_params
from teflon.util.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

layers = tf.keras.layers
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def distribute_value(value, num_proc):
    """Adjusts training parameters for distributed training.
    In case of distributed training frequencies expressed in global steps have
    to be adjusted to local steps, thus divided by the number of processes.
    """
    return max(value // num_proc, 1)

def mlp(hidden_sizes=(64, 32), activation='relu', output_activation=None,
        layer_norm=False):
    """Creates MLP with the specified parameters."""
    model = tf.keras.Sequential()

    for h in hidden_sizes[:-1]:
        model.add(tf.keras.layers.Dense(units=h, activation=activation))  # modify
        if layer_norm:
            model.add(tf.keras.layers.LayerNormalization())
        model.add(tf.keras.layers.Activation(activation))

    model.add(tf.keras.layers.Dense(units=hidden_sizes[-1], activation=activation))  # modify
    if layer_norm:
        model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.Activation(output_activation))

    return model

@tf.function
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def make_actor_discrete(observation_space, action_space, hidden_sizes,
                        activation, layer_norm):
    """Creates actor tf.keras.Model.
    This function can be used only in environments with discrete action space.
    """

    class DiscreteActor(tf.keras.Model):
        """Actor model for discrete action space."""

        def __init__(self, observation_space, action_space, hidden_sizes,
                     activation, layer_norm):
            super().__init__()
            self._act_dim = action_space.n

            obs_input = tf.keras.Input(shape=observation_space.shape)
            actor = mlp(
                hidden_sizes=hidden_sizes + [action_space.n],
                activation=activation,
                layer_norm=layer_norm
            )(obs_input)

            self._network = tf.keras.Model(inputs=obs_input, outputs=actor)

        @tf.function
        def call(self, inputs, training=None, mask=None):
            return tf.nn.log_softmax(self._network(inputs))

        @tf.function
        def action(self, observations):
            return tf.squeeze(tf.random.categorical(self(observations), 1),
                              axis=1)

        @tf.function
        def action_logprob(self, observations, actions):
            return tf.reduce_sum(
                tf.math.multiply(self(observations),
                                 tf.one_hot(tf.cast(actions, tf.int32),
                                            depth=self._act_dim)), axis=-1)

    return DiscreteActor(observation_space, action_space, hidden_sizes,
                         activation, layer_norm)


def make_actor_continuous(action_space, hidden_sizes,
                          activation, layer_norm):
    """Creates actor tf.keras.Model.
    This function can be used only in environments with continuous action space.
    """

    class ContinuousActor(tf.keras.Model):
        """Actor model for continuous action space."""

        def __init__(self, action_space, hidden_sizes,
                     activation, layer_norm):
            super().__init__()
            self._action_dim = action_space.shape

            self._mu = mlp(
                hidden_sizes=hidden_sizes + [self._action_dim[0]],  # modify
                activation=activation,
                layer_norm=layer_norm
            )
            self._log_std = tf.Variable(
                initial_value=-0.5 * np.ones(shape=(1,) + self._action_dim,
                                             dtype=np.float32), trainable=True,
                name='log_std_dev')

        @tf.function
        def call(self, inputs, training=None, mask=None):
            mu = self._mu(inputs)
            log_std = tf.clip_by_value(self._log_std, LOG_STD_MIN, LOG_STD_MAX)

            return mu, log_std

        @tf.function
        def action(self, observations):
            mu, log_std = self(observations)
            std = tf.exp(log_std)
            return mu + tf.random.normal(tf.shape(input=mu)) * std

        @tf.function
        def action_logprob(self, observations, actions):
            mu, log_std = self(observations)
            return gaussian_likelihood(actions, mu, log_std)

    return ContinuousActor(action_space, hidden_sizes,
                           activation, layer_norm)


def make_critic(observation_space, hidden_sizes, activation):
    """Creates critic tf.keras.Model"""
    obs_input = tf.keras.Input(shape=observation_space.shape)

    critic = tf.keras.Sequential([
        mlp(hidden_sizes=hidden_sizes + [1],
            activation=activation),
        tf.keras.layers.Reshape([]),
    ])(obs_input)

    return tf.keras.Model(inputs=obs_input, outputs=critic)


def mlp_actor_critic(observation_space, action_space, hidden_sizes=[64,64],
                     activation=tf.tanh, layer_norm=False):
    """Creates actor and critic tf.keras.Model-s."""
    actor = None

    # default policy builder depends on action space
    if isinstance(action_space, gym.spaces.Discrete):
        actor = make_actor_discrete(observation_space, action_space,
                                    hidden_sizes,
                                    activation, layer_norm)
    elif isinstance(action_space, gym.spaces.Box):
        actor = make_actor_continuous(action_space,
                                      hidden_sizes,
                                      activation, layer_norm)

    critic = make_critic(observation_space, hidden_sizes, activation)

    return actor, critic


class PPO(tf.keras.Model):
    def __init__(
            self,
            # state_dim,
            # action_dim,
            # max_action,
            feature_extractor,
            observation_space,
            action_space,
            steps_per_epoch=4000, 
            epochs=50, 
            gamma=0.99, 
            clip_ratio=0.2, 
            pi_lr=3e-4, 
            vf_lr=1e-3, 
            gpu=0):
        super().__init__()
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        self.actor, self.critic = mlp_actor_critic(observation_space, action_space)
        obs_dim = observation_space.shape[0]
        # self.actor.build(input_shape=(None, obs_dim))
        # self.critic.build(input_shape=(None, obs_dim))
        self.actor.build(input_shape=(None, obs_dim))
        self.critic.build(input_shape=(None, obs_dim))

        self.actor_optimizer = MpiAdamOptimizer(learning_rate=pi_lr)
        self.critic_optimizer = MpiAdamOptimizer(learning_rate=vf_lr)

        sync_params(self.actor.variables)
        sync_params(self.critic.variables)

        self.ofe_net = feature_extractor

    @tf.function
    def value(self, observations):
        return self.critic(observations)

    def get_value(self, observation):
        return self.value(np.array([observation])).numpy()[0]

    @tf.function
    def value_loss(self, observations, rtg):
        return tf.reduce_mean((self.critic(observations) - rtg) ** 2)

    @tf.function
    def value_train_step(self, observations, rtg):
        def loss():
            return self.value_loss(observations, rtg)

        self.critic_optimizer.minimize(loss, self.critic.trainable_variables)
        sync_params(self.critic.variables)

        return loss()

    def get_action(self, observation):
        return self.actor.action(np.array([observation])).numpy()[0]

    @tf.function
    def pi_loss(self, logp, logp_old, advantages):
        ratio = tf.exp(logp - logp_old)
        min_adv = tf.where(condition=(advantages >= 0),
                           x=(1 + self.clip_ratio) * advantages,
                           y=(1 - self.clip_ratio) * advantages)
        tf.summary.scalar(name="PPO/ratio", data=tf.reduce_mean(ratio))
        return -tf.reduce_mean(tf.minimum(ratio * advantages, min_adv))

    @tf.function
    def pi_train_step(self, observations, actions, advantages, logp_old):
        def loss():
            logp = self.actor.action_logprob(observations, actions)
            return self.pi_loss(logp, logp_old, advantages)

        self.actor_optimizer.minimize(loss, self.actor.trainable_variables)
        sync_params(self.actor.variables)

        # For logging purposes
        logp = self.actor.action_logprob(observations, actions)
        loss_new = self.pi_loss(logp, logp_old, advantages)

        return loss_new, tf.reduce_mean(logp_old - logp), tf.reduce_mean(-logp)

    def train(self, replay_buffer, train_pi_iters=80, train_v_iters=80, lam=0.97, target_kl=0.01):
        [batch_obs, batch_act, batch_adv, batch_rtg, batch_logp] = replay_buffer.get()

        for i in range(train_pi_iters):
            actor_loss, kl, entropy = self.pi_train_step(batch_obs, batch_act,
                                                batch_adv, batch_logp)

            kl = mpi_avg(kl)
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break

        for _ in range(train_v_iters):
            critic_loss = self.value_train_step(batch_obs, batch_rtg)

        # Visualize results in TensorBoard
        tf.summary.scalar(name="PPO/actor_loss", data=actor_loss)
        # tf.summary.scalar(name="PPO/logp_max",
        #                   data=np.max(logp_news))
        # tf.summary.scalar(name="PPO/logp_min",
        #                   data=np.min(logp_news))
        tf.summary.scalar(name="PPO/entropy", data=entropy)
        # tf.summary.scalar(name="PPO/adv_max",
        #                   data=np.max(advantages))
        # tf.summary.scalar(name="PPO/adv_min",
        #                   data=np.min(advantages))
        tf.summary.scalar(name="PPO/kl", data=kl)
        tf.summary.scalar(name="PPO/critic_loss", data=critic_loss)
        return actor_loss, critic_loss

    def save(self, save_dir):
        self.actor.save_weights(os.path.join(save_dir,'agent_actor_model'))
        self.critic.save_weights(os.path.join(save_dir,'agent_critic_model'))

    def load(self, load_dir):
        self.actor.load_weights(os.path.join(load_dir,'agent_actor_model'))
        self.critic.load_weights(os.path.join(load_dir,'agent_critic_model'))
