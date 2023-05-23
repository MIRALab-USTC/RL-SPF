# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

# Forked from https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/ppo.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
import os

layers = tf.keras.layers
EPS = 1e-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class GaussianActor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action, layer_units=(256, 256),
                 hidden_activation="tanh", name='gaussian_policy'):
        super().__init__(name=name)
        initializer = tf.keras.initializers.Orthogonal()
        base_layers = []
        for cur_layer_size in layer_units:
            cur_layer = layers.Dense(cur_layer_size, activation=hidden_activation, kernel_initializer=initializer)
            base_layers.append(cur_layer)

        self.base_layers = base_layers

        self.out_mean = layers.Dense(action_dim, name="L_mean", kernel_initializer=initializer)
        # State independent log covariance
        self.out_logstd = tf.Variable(
            initial_value=-0.5 * np.ones(action_dim, dtype=np.float32),
            dtype=tf.float32, name="logstd")

        self._max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        self(dummy_state)

    def _dist_from_states(self, states):
        features = states

        for cur_layer in self.base_layers:
            features = cur_layer(features)

        mu_t = self.out_mean(features)

        log_sigma_t = tf.clip_by_value(self.out_logstd, LOG_STD_MIN, LOG_STD_MAX)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mu_t, scale_diag=tf.exp(log_sigma_t))

        return dist

    def call(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        # actions = raw_actions * self._max_action
        return raw_actions, log_pis

    def mean_action(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.mean()
        log_pis = dist.log_prob(raw_actions)

        # actions = raw_actions * self._max_action
        return raw_actions, log_pis

    def compute_log_probs(self, states, actions):
        dist = self._dist_from_states(states)
        raw_actions = actions

        # raw_actions = actions / self._max_action
        log_pis = dist.log_prob(raw_actions)

        return log_pis


class CriticV(tf.keras.Model):
    def __init__(self, state_dim, units, name='qf'):
        super().__init__(name=name)

        initializer = tf.keras.initializers.Orthogonal()
        self.l1 = layers.Dense(units[0], name="L1", activation='tanh', kernel_initializer=initializer)
        self.l2 = layers.Dense(units[1], name="L2", activation='tanh', kernel_initializer=initializer)
        # self.l3 = Dense(1, name="L2", activation='linear')
        self.l3 = layers.Dense(1, name="L3", kernel_initializer=initializer)

        with tf.device('/cpu:0'):
            self(tf.constant(np.zeros(shape=(1, state_dim), dtype=np.float32)))

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)


class PPO(tf.keras.Model):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            feature_extractor,
            actor_units=(64, 64),
            critic_units=(64, 64),
            pi_lr=3e-4,
            vf_lr=1e-3,
            clip_ratio=0.2,
            batch_size=64,
            discount=0.99,
            n_epoch=10,
            horizon=2048,
            gpu=0):
        super().__init__()
        self.batch_size = batch_size
        self.discount = discount
        self.n_epoch = n_epoch
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        self.horizon = horizon
        self.clip_ratio = clip_ratio
        assert self.horizon % self.batch_size == 0, \
            "Horizon should be divisible by batch size"

        self.actor = GaussianActor(
            feature_extractor.dim_state_features,
            action_dim, max_action, actor_units)
        self.critic = CriticV(
            feature_extractor.dim_state_features, critic_units)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=pi_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=vf_lr)

        self.ofe_net = feature_extractor

    def get_action(self, raw_state, test=False):
        assert isinstance(raw_state, np.ndarray), \
            "Input instance should be np.ndarray, not {}".format(type(raw_state))

        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        action, logp = self._get_action_body(raw_state, test)[:2]

        if is_single_input:
            return action.numpy()[0], logp.numpy()
        else:
            return action.numpy(), logp.numpy()

    def get_action_and_val(self, raw_state, test=False):
        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        action, logp, v = self._get_action_logp_v_body(raw_state, test)

        if is_single_input:
            v = v[0]
            action = action[0]

        return action.numpy(), logp.numpy(), v.numpy()

    def get_logp_and_val(self, raw_state, action):
        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        state_feature = self.ofe_net.features_from_states(raw_state)
        logp = self.actor.compute_log_probs(state_feature, action)  
        v = self.critic(state_feature)

        if is_single_input:
            v = v[0]
            action = action[0]

        return logp.numpy(), v.numpy()

    def get_val(self, raw_state):
        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        state_feature = self.ofe_net.features_from_states(raw_state)
        v = self.critic(state_feature)

        if is_single_input:
            v = v[0]

        return v.numpy()

    @tf.function
    def _get_action_logp_v_body(self, raw_state, test):
        action, logp = self._get_action_body(raw_state, test)[:2]
        state_feature = self.ofe_net.features_from_states(raw_state)
        v = self.critic(state_feature)
        return action, logp, v

    @tf.function
    def _get_action_body(self, state, test):
        state_feature = self.ofe_net.features_from_states(state)
        if test:
            return self.actor.mean_action(state_feature)
        else:
            return self.actor(state_feature)

    def select_action(self, raw_state):
        action, logp = self.get_action(raw_state, test=True)
        return action

    def train(self, replay_buffer, train_pi_iters=80, train_v_iters=80, target_kl=0.01):
        [raw_states, actions, advantages, returns, logp_olds] = replay_buffer.get()
        # Train actor and critic
        for i in range(train_pi_iters):
            actor_loss, kl, entropy, logp_news, ratio = self._train_actor_body(
                raw_states, actions, advantages, logp_olds)
            if kl > 1.5 * target_kl:
                print('Early stopping at step %d due to reaching max kl.' % i)
                break
        for _ in range(train_v_iters):
            critic_loss = self._train_critic_body(raw_states, returns)

        # Visualize results in TensorBoard
        tf.summary.scalar(name="PPO/actor_loss", data=actor_loss)
        # tf.summary.scalar(name="PPO/logp_max",
        #                   data=np.max(logp_news))
        # tf.summary.scalar(name="PPO/logp_min",
        #                   data=np.min(logp_news))
        tf.summary.scalar(name="PPO/logp_mean", data=np.mean(logp_news))
        # tf.summary.scalar(name="PPO/adv_max",
        #                   data=np.max(advantages))
        # tf.summary.scalar(name="PPO/adv_min",
        #                   data=np.min(advantages))
        tf.summary.scalar(name="PPO/kl", data=kl)
        tf.summary.scalar(name="PPO/entropy", data=entropy)
        tf.summary.scalar(name="PPO/ratio", data=tf.reduce_mean(ratio))
        tf.summary.scalar(name="PPO/critic_loss", data=critic_loss)
        return actor_loss, critic_loss

    @tf.function
    def _train_actor_body(self, raw_states, actions, advantages, logp_olds):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                state_features = self.ofe_net.features_from_states(raw_states)
                logp_news = self.actor.compute_log_probs(
                    state_features, actions)
                ratio = tf.math.exp(logp_news - tf.squeeze(logp_olds))
                # min_adv = tf.clip_by_value(
                #     ratio,
                #     1.0 - self.clip_ratio,
                #     1.0 + self.clip_ratio) * tf.squeeze(advantages)
                min_adv = tf.where(condition=(advantages >= 0),
                           x=(1 + self.clip_ratio) * advantages,
                           y=(1 - self.clip_ratio) * advantages)
                actor_loss = -tf.reduce_mean(tf.minimum(
                    ratio * tf.squeeze(advantages),
                    min_adv))
            actor_grad = tape.gradient(
                actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(actor_grad, self.actor.trainable_variables))
        kl = tf.reduce_mean(tf.squeeze(logp_olds) - logp_news)
        entropy = tf.reduce_mean(-logp_news)

        # return actor_loss, logp_news, ratio
        return actor_loss, kl, entropy, logp_news, ratio

    @tf.function
    def _train_critic_body(self, raw_states, returns):
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                state_features = self.ofe_net.features_from_states(raw_states)
                current_V = self.critic(state_features)
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(0.5 * tf.square(td_errors))
            critic_grad = tape.gradient(
                critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(
                zip(critic_grad, self.critic.trainable_variables))

        return critic_loss

    def save(self, save_dir):
        self.actor.save_weights(os.path.join(save_dir,'agent_actor_model'))
        self.critic.save_weights(os.path.join(save_dir,'agent_critic_model'))

    def load(self, load_dir):
        self.actor.load_weights(os.path.join(load_dir,'agent_actor_model'))
        self.critic.load_weights(os.path.join(load_dir,'agent_critic_model'))
