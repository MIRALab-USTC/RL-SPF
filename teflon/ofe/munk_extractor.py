# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import logging

import gin
import numpy as np
import tensorflow as tf

layers = tf.keras.layers


@gin.configurable
class MunkNet(tf.keras.Model):
    def __init__(self, dim_state, dim_action,
                 internal_states="small",
                 hidden_units=100,
                 lam_m=10,
                 activation=tf.nn.relu, name='MunkNet'):
        super().__init__(name=name)
        self.lam_m = lam_m
        self.act = activation

        if internal_states == "small":
            internal_states = dim_state // 3
        elif internal_states == "big":
            internal_states = dim_state
        elif isinstance(internal_states, int):
            internal_states = dim_state + int(internal_states)
        else:
            raise ValueError("invalid internal state: {}".format(internal_states))

        self.internal_states = internal_states

        logger = logging.getLogger(__name__)
        print("state variable is {}".format(internal_states))
        self.state_layer = layers.Dense(internal_states, name="state_feature")

        # predict next state
        self.hidden_layer = layers.Dense(hidden_units, name="next_state_feature")
        self.state_linear = layers.Dense(internal_states, name="linear_state")
        self.reward_linear = layers.Dense(1, name="linear_reward")

        self.aux_optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        dummy_state = tf.constant(np.zeros(shape=[1, dim_state], dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, dim_action], dtype=np.float32))
        self([dummy_state, dummy_action])
        self._dim_state_features = int(self.features_from_states(dummy_state).shape[1])
        self._dim_state_action_features = int(self.features_from_states_actions(dummy_state, dummy_action).shape[1])

    @property
    def dim_state_features(self):
        return self._dim_state_features

    @property
    def dim_state_action_features(self):
        return self._dim_state_action_features

    def features_from_states(self, states):
        features = self.state_layer(states)
        return features

    def features_from_states_actions(self, states, actions):
        state_features = self.features_from_states(states)
        features = tf.concat([state_features, actions], axis=1)
        return features

    def call(self, inputs):
        [states, actions] = inputs
        sa_features = self.features_from_states_actions(states, actions)
        features = self.hidden_layer(sa_features)
        features = tf.nn.relu(features)

        predict_next_states = self.state_linear(features)
        predict_next_rewards = self.reward_linear(features)

        return predict_next_states, predict_next_rewards

    @tf.function
    def train(self, obs, actions, next_obs, rewards, dones):
        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                predicted_next_states, predict_rewards = self([obs, actions])
                next_states = tf.stop_gradient(self.features_from_states(next_obs))

                feature_loss = tf.reduce_sum((predicted_next_states - next_states) ** 2) + \
                               self.lam_m * tf.reduce_sum((rewards - predict_rewards) ** 2)

            feature_grad = tape.gradient(feature_loss, self.trainable_variables)
            self.aux_optimizer.apply_gradients(zip(feature_grad, self.trainable_variables))

        tf.summary.scalar(name="ofe/uniform_ofe_loss", data=feature_loss)
