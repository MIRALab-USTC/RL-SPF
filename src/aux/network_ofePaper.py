# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import logging

import gin
import numpy as np
import tensorflow as tf

from src.aux.blocks import ResnetBlock, DensenetBlock, MLPBlock

layers = tf.keras.layers


@gin.configurable
class OFENet(tf.keras.Model):
    def __init__(self, dim_state, dim_action, dim_output, total_units,
                 num_layers, batchnorm, activation=tf.nn.relu, block="normal",
                 kernel_initializer="glorot_uniform",
                 trainable=True, name='FeatureNet',
                 gpu=0, skip_action_branch=False):
        super().__init__(name=name)
        self._gpu = gpu
        self._skip_action_branch = skip_action_branch

        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(fmt='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s',
                                               datefmt="%m/%d %I:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        state_layer_units, action_layer_units = calculate_layer_units(dim_state, dim_action, block, total_units,
                                                                      num_layers)
        self.act = activation
        self.batchnorm = batchnorm

        if block not in ["resnet", "densenet", "normal"]:
            raise ValueError("invalid connect : {}".format(block))

        state_blocks = []
        action_blocks = []

        if block == "resnet":
            assert len(state_layer_units) % 2 == 0
            assert len(action_layer_units) % 2 == 0
            block_class = ResnetBlock

            for i in range(len(state_layer_units) // 2):
                cur_block = block_class(units1=state_layer_units[i * 2], units2=state_layer_units[i * 2 + 1],
                                        activation=self.act,
                                        kernel_initializer=kernel_initializer,
                                        batchnorm=batchnorm,
                                        trainable=trainable)
                state_blocks.append(cur_block)

            for i in range(len(action_layer_units) // 2):
                cur_block = block_class(units1=action_layer_units[i * 2], units2=action_layer_units[i * 2 + 1],
                                        activation=self.act,
                                        kernel_initializer=kernel_initializer,
                                        batchnorm=batchnorm,
                                        trainable=trainable)
                action_blocks.append(cur_block)
        else:
            if block == "densenet":
                block_class = DensenetBlock
            elif block == "normal":
                block_class = MLPBlock
            else:
                raise ValueError("invalid block {}".format(block))

            for idx_layer, cur_layer_units in enumerate(state_layer_units):
                cur_block = block_class(units=cur_layer_units, activation=self.act,
                                        kernel_initializer=kernel_initializer,
                                        batchnorm=batchnorm, trainable=trainable,
                                        name="state{}".format(idx_layer))
                state_blocks.append(cur_block)

            for idx_layer, cur_layer_units in enumerate(action_layer_units):
                cur_block = block_class(units=cur_layer_units, activation=self.act,
                                        kernel_initializer=kernel_initializer,
                                        batchnorm=batchnorm, trainable=trainable,
                                        name="action{}".format(idx_layer))
                action_blocks.append(cur_block)

        self.state_blocks = state_blocks
        self.action_blocks = action_blocks

        self.out_layer = layers.Dense(dim_output,
                                      trainable=trainable,
                                      name="feat_out")

        self.output_dim = dim_output

        if trainable:
            self.aux_optimizer = tf.optimizers.Adam(learning_rate=3e-4)
        else:
            self.aux_optimizer = None

        dummy_state = tf.constant(np.zeros(shape=[1, dim_state], dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, dim_action], dtype=np.float32))
        self([dummy_state, dummy_action])
        self._dim_state_features = int(self.features_from_states(dummy_state).shape[1])
        self._dim_state_action_features = int(self.features_from_states_actions(dummy_state, dummy_action).shape[1])

        add_dim = self.dim_state_features - dim_state
        logger.debug("state feature dim is {} (state-dim: {}, add: {})".format(
            self.dim_state_features, dim_state, add_dim))
        add_dim = self.dim_state_action_features - (dim_action + self.dim_state_features)
        logger.debug("state-action feature dim is {} (action-dim: {}, add: {})".format(
            self.dim_state_action_features, dim_action, add_dim))

        import ipdb
        ipdb.set_trace

    @property
    def dim_state_features(self):
        return self._dim_state_features

    @property
    def dim_state_action_features(self):
        return self._dim_state_action_features

    def call(self, inputs):
        [states, actions] = inputs

        features = states

        for cur_block in self.state_blocks:
            features = cur_block(features, training=True)

        if not self._skip_action_branch:
            features = tf.concat([features, actions], axis=1)

            for cur_block in self.action_blocks:
                features = cur_block(features, training=True)

        values = self.out_layer(features)
        return values

    def features_from_states(self, states):
        features = states
        for cur_block in self.state_blocks:
            features = cur_block(features, training=False)

        return features

    def features_from_states_actions(self, states, actions):
        state_features = self.features_from_states(states)
        features = tf.concat([state_features, actions], axis=1)

        for cur_block in self.action_blocks:
            features = cur_block(features, training=False)

        return features

    @tf.function
    def train(self, states, actions, next_states, rewards, dones):
        with tf.device("/gpu:{}".format(self._gpu)):
            with tf.GradientTape() as tape:
                predicted_states = self([states, actions])

                target_dim = self.output_dim
                target_states = next_states[:, :target_dim]
                feature_loss = tf.reduce_mean((target_states - predicted_states) ** 2)

            feature_grad = tape.gradient(feature_loss, self.trainable_variables)
            self.aux_optimizer.apply_gradients(zip(feature_grad, self.trainable_variables))


def calculate_layer_units(state_dim, action_dim, ofe_block, total_units, num_layers):
    assert total_units % num_layers == 0

    if ofe_block == "densenet":
        per_unit = total_units // num_layers
        state_layer_units = [per_unit] * num_layers
        action_layer_units = [per_unit] * num_layers

    elif ofe_block in ["normal", "resnet"]:
        state_layer_units = [total_units + state_dim] * num_layers
        action_layer_units = [total_units * 2 + state_dim + action_dim] * num_layers
    else:
        raise ValueError("invalid connection type")

    return state_layer_units, action_layer_units
