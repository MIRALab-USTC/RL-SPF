# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

"""
DDPG implementation in Tensorflow Eager Execution

This implementation follows the setting in
 https://github.com/sfujim/TD3/blob/master/OurDDPG.py

mini-batch size is 256 instead of 100, which is written in the above paper.

"""

import os
import numpy as np
import tensorflow as tf
import trfl.target_update_ops as target_update
from tensorflow.python.training.tracking import tracking

layers = tf.keras.layers
regularizers = tf.keras.regularizers
losses = tf.keras.losses


class Actor(tf.keras.Model):
    def __init__(self, state_feature_dim, action_dim, max_action, name="Actor"):
        super().__init__(name=name)

        self.l1 = layers.Dense(400, name="L1")
        self.l2 = layers.Dense(300, name="L2")
        self.l3 = layers.Dense(action_dim, name="L3")

        self.max_action = max_action

        dummy_features = tf.constant(np.zeros(shape=[1, state_feature_dim], dtype=np.float32))
        self(dummy_features)

    def call(self, inputs):
        with tf.device("/gpu:0"):
            features = tf.nn.relu(self.l1(inputs))
            features = tf.nn.relu(self.l2(features))
            features = self.l3(features)
            action = self.max_action * tf.nn.tanh(features)
        return action


class Critic(tf.keras.Model):
    def __init__(self, state_action_feature_dim, name="Critic"):
        super().__init__(name=name)

        self.l1 = layers.Dense(400, name="L1")
        self.l2 = layers.Dense(300, name="L2")
        self.l3 = layers.Dense(1, name="L3")

        dummy_features = tf.constant(np.zeros(shape=[1, state_action_feature_dim], dtype=np.float32))
        self(dummy_features)

    def call(self, inputs):
        with tf.device("/gpu:0"):
            features = tf.nn.relu(self.l1(inputs))
            features = tf.nn.relu(self.l2(features))
            features = self.l3(features)

        return features


class DDPG(tracking.AutoTrackable):
    def __init__(self, state_dim, action_dim, max_action, feature_extractor, action_noise=0.1):
        self._action_noise = action_noise
        self._extractor = feature_extractor

        self.actor = Actor(self._extractor.dim_state_features, action_dim, max_action)
        self.actor_target = Actor(self._extractor.dim_state_features, action_dim, max_action)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        # initialize target network
        for param, target_param in zip(self.actor.weights, self.actor_target.weights):
            target_param.assign(param)

        self.critic = Critic(self._extractor.dim_state_action_features)
        self.critic_target = Critic(self._extractor.dim_state_action_features)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        # initialize target network
        for param, target_param in zip(self.critic.weights, self.critic_target.weights):
            target_param.assign(param)

    def select_action(self, state):
        """

        :param np.ndarray state:
        :return:
        """
        assert isinstance(state, np.ndarray)
        state =  state.astype(np.float32)
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
            action = self._select_action_body(tf.constant(state))
            action = tf.squeeze(action, axis=0)
        else:
            action = self._select_action_body(state)

        return action.numpy()

    @tf.function
    def _select_action_body(self, states):
        """

        :param np.ndarray states:
        :return:
        """
        features = self._extractor.features_from_states(states)
        action = self.actor(features)
        return action

    def select_action_noise(self, state):
        """

        :param np.ndarray state:
        :return:
        """
        assert isinstance(state, np.ndarray)
        state =  state.astype(np.float32)
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
            action = self._select_action_body(tf.constant(state))
            action = action + tf.random.normal(shape=action.shape, stddev=self._action_noise)
            # action = tf.clip_by_value(
            #     action + tf.random.normal(shape=action.shape, stddev=self._action_noise*self._max_action), \
            #     -self._max_action, self._max_action)
            action = tf.squeeze(action, axis=0)
        else:
            action = self._select_action_body(state)
            action = action + tf.random.normal(shape=action.shape, stddev=self._action_noise)

        return action.numpy()


    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005):
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        done = np.array(done, dtype=np.float32)
        critic_loss, actor_loss = self._train_body(state, action, next_state, reward, done, discount, tau)

        tf.summary.scalar(name="loss/ActorLoss", data=actor_loss)
        tf.summary.scalar(name="loss/CriticLoss", data=critic_loss)

        return actor_loss, critic_loss

    @tf.function
    def _train_body(self, states, actions, next_states, rewards, done, discount, tau):
        with tf.device("/gpu:0"):
            not_done = 1 - done

            with tf.GradientTape() as tape:
                next_states_features = self._extractor.features_from_states(next_states)
                target_sa_feats = self._extractor.features_from_states_actions(
                    states=next_states,
                    actions=self.actor_target(next_states_features))

                target_Q = self.critic_target(target_sa_feats)
                target_Q = rewards + (not_done * discount * target_Q)
                # detach => stop_gradient
                target_Q = tf.stop_gradient(target_Q)

                sa_feats = self._extractor.features_from_states_actions(states=states, actions=actions)
                current_Q = self.critic(sa_feats)

                # Compute critic loss
                critic_loss = tf.reduce_mean(losses.MSE(current_Q, target_Q))

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

            with tf.GradientTape() as tape:
                a_feats = self._extractor.features_from_states(states)
                policy_action = self.actor(a_feats)
                sa_policy_feats = self._extractor.features_from_states_actions(states, policy_action)
                actor_loss = -tf.reduce_mean(self.critic(sa_policy_feats))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            target_update.update_target_variables(self.critic_target.weights, self.critic.weights, tau)
            target_update.update_target_variables(self.actor_target.weights, self.actor.weights, tau)

            return actor_loss, critic_loss


    def save(self, save_dir):
        self.actor.save_weights(os.path.join(save_dir,'agent_actor_model'))
        self.actor_target.save_weights(os.path.join(save_dir,'agent_actor_target_model'))
        self.critic.save_weights(os.path.join(save_dir,'agent_critic_model'))
        self.critic_target.save_weights(os.path.join(save_dir,'agent_critic_target_model'))

    def load(self, load_dir):
        self.actor.load_weights(os.path.join(load_dir,'agent_actor_model'))
        self.actor_target.load_weights(os.path.join(load_dir,'agent_actor_target_model'))
        self.critic.load_weights(os.path.join(load_dir,'agent_critic_model'))
        self.critic_target.load_weights(os.path.join(load_dir,'agent_critic_target_model'))
