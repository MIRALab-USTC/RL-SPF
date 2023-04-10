# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import logging

import gin
import numpy as np
import tensorflow as tf
import ipdb
import os
import pandas as pd
import matplotlib.pyplot as plt
import math

from teflon.ofe.blocks import ResnetBlock, DensenetBlock, MLPBlock


layers = tf.keras.layers


@gin.configurable
class OFENet(tf.keras.Model):
    def __init__(self, dim_state, dim_action, dim_discretize, total_units,
                 num_layers, batchnorm, 
                 fourier_type, discount,  
                 use_projection=True, projection_dim=256, cosine_similarity=True, 
                 activation=tf.nn.relu, block="densenet",
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
        self.block = block

        if block not in ["densenet", "mlp","mlp_cat"]:
            raise ValueError("invalid connect : {}".format(block))

        state_blocks = []
        action_blocks = []

        if block == "densenet":
            block_class = DensenetBlock
        elif block == "mlp" or "mlp_cat":
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
        
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_discretize = dim_discretize

        self.end = int(dim_discretize*0.5 + 1)  # predict fourier function in [0,\pi]
        self.prediction = Prediction(dim_discretize=self.end, dim_state=dim_state)

        if use_projection == True:
            self.projection = Projection(classifier_type='mlp', output_dim=projection_dim)
            self.projection2 = Projection2(classifier_type='mlp', output_dim=projection_dim)

        self.fourier_type = fourier_type
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.cosine_similarity = cosine_similarity

        if trainable:
            self.aux_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        else:
            self.aux_optimizer = None

        dummy_state = tf.constant(np.zeros(shape=[1, dim_state], dtype=np.float32))
        dummy_action = tf.constant(np.zeros(shape=[1, dim_action], dtype=np.float32))
        self([dummy_state, dummy_action])
        self._dim_state_features = int(self.features_from_states(dummy_state).shape[1])  # 输出output feature的列数，即为output的维度
        self._dim_state_action_features = int(self.features_from_states_actions(dummy_state, dummy_action).shape[1])

        # add_dim: dim(state_feature) - dim(state)
        add_dim = self.dim_state_features - dim_state
        logger.debug("state feature dim is {} (state-dim: {}, add: {})".format(
            self.dim_state_features, dim_state, add_dim))
        # add_dim: dim(action_feature) - dim(state)
        add_dim = self.dim_state_action_features - (dim_action + self.dim_state_features)
        logger.debug("state-action feature dim is {} (action-dim: {}, add: {})".format(
            self.dim_state_action_features, dim_action, add_dim))

        ratio = 2 * np.pi / dim_discretize
        con = tf.constant([k * ratio for k in range(self.end)])
        self.Gamma_re = tf.constant(discount * np.diag(tf.math.cos(con)))
        self.Gamma_im = tf.constant(-discount * np.diag(tf.math.sin(con)))

        ratio2 = 1.0/(2*np.pi*dim_discretize)
        con_re = tf.constant([1] + [2*math.cos(k * ratio) for k in range(1, self.end-1)] + [-1])
        self.con_re = tf.expand_dims(ratio2 * con_re, axis = 0)
        con_im = tf.constant([0] + [-2*math.sin(k * ratio) for k in range(1, self.end-1)] + [0])
        self.con_im = tf.expand_dims(ratio2 * con_im, axis = 0)
        

        if fourier_type == 'dft':
            ratio = 2 * np.pi * (dim_discretize - 1) / dim_discretize
            con = tf.constant([k * ratio for k in range(dim_discretize)])
            self.con_re = tf.constant(tf.math.cos(con))
            self.con_im = tf.constant(tf.math.sin(con))
            self.coef = pow(discount, dim_discretize - 1)


    @property
    def dim_state_features(self):
        return self._dim_state_features

    @property
    def dim_state_action_features(self):
        return self._dim_state_action_features

    def call(self, inputs, training=True):

        [states, actions] = inputs
        batch_size = states.shape[0]

        features = states

        for cur_block in self.state_blocks:
            features = cur_block(features, training=training)
        if self.block == "mlp_cat":
            features = tf.concat([features, states], axis=1)

        if not self._skip_action_branch:
            features = tf.concat([features, actions], axis=1)

            for cur_block in self.action_blocks:
                features = cur_block(features, training=training)

            if self.block == "mlp_cat":
                features = tf.concat([features, states, actions], axis=1)

        # predictor layer
        predictor_re, predictor_im = self.prediction(features, training=training)
        predictor_re = tf.reshape(predictor_re, [batch_size, self.end, self.dim_state])
        predictor_im = tf.reshape(predictor_im, [batch_size, self.end, self.dim_state])
        # predictor_re = tf.reshape(self.out_layer_re(features), [batch_size, self.dim_discretize, self.dim_state])
        # predictor_im = tf.reshape(self.out_layer_im(features), [batch_size, self.dim_discretize, self.dim_state])

        return predictor_re, predictor_im

    def features_from_states(self, states):
        features = states
        for cur_block in self.state_blocks:
            features = cur_block(features, training=False)
        
        if self.block == "mlp_cat":
            features = tf.concat([features, states], axis=1)

        return features

    def features_from_states_actions(self, states, actions):
        state_features = self.features_from_states(states)
        features = tf.concat([state_features, actions], axis=1)

        for cur_block in self.action_blocks:
            features = cur_block(features, training=False)

        if self.block == "mlp_cat":
            features = tf.concat([features, states, actions], axis=1)

        return features

    def loss(self, y_target, y, target_model=None):
        trun = 15

        if self.use_projection == True and target_model is not None:

            y_target2 = target_model.projection(y_target[:,trun:self.end-trun,:])  # add the weights of projection to the weights of OFENet
            con = target_model.projection2(y_target2)  # no use, just for adding projection2 to extractor_target's structure
            y2 = self.projection(y[:,trun:self.end-trun,:])
            y2 = self.projection2(y2)
        
        if self.cosine_similarity == True:
            loss_fun = tf.keras.losses.CosineSimilarity(axis=-1)
            loss1 = loss_fun(y_target[:,0:trun,:], y[:,0:trun,:])
            loss2 = loss_fun(y_target2, y2)
            loss3 = loss_fun(y_target[:,self.end-trun:self.end,:], y[:,self.end-trun:self.end,:])
            loss = loss1 + loss2 + loss3
        else:
            loss = tf.nn.l2_loss(y_target2 - y2)

        return loss


    @tf.function
    def train(self, target_model, states, actions, next_states, next_actions, dones, Hth_states = None):
        
        with tf.device("/gpu:{}".format(self._gpu)):
            with tf.GradientTape(persistent=True) as tape:

                dones = tf.cast(dones, dtype=tf.float32)
                dones = tf.tile(tf.expand_dims(dones, axis = -1), multiples=[1, self.end, self.dim_state])
                O = tf.tile(tf.expand_dims(next_states, axis = 1), multiples=[1, self.end, 1])
                [predicted_re, predicted_im] = self([states, actions])
                [next_predicted_re, next_predicted_im] = target_model([next_states, next_actions])

                if self.fourier_type == 'dtft':

                    y_target_re = tf.stop_gradient(O + (tf.matmul(self.Gamma_re, next_predicted_re) - tf.matmul(self.Gamma_im, next_predicted_im))*(1 - dones))
                    y_re = predicted_re
                    pred_re_loss = self.loss(y_target_re, y_re, target_model)

                    y_target_im = tf.stop_gradient((tf.matmul(self.Gamma_im, next_predicted_re) + tf.matmul(self.Gamma_re, next_predicted_im))*(1 - dones))
                    y_im = predicted_im
                    pred_im_loss = self.loss(y_target_im, y_im, target_model)

                    # inv_s2 = tf.squeeze(tf.matmul(self.con_re, predicted_re) + tf.matmul(self.con_im, predicted_im), axis=1)
                    # loss_fun = tf.keras.losses.CosineSimilarity(axis=-1)
                    # inv_loss = loss_fun(inv_s2, next_states)
                    # inv_loss = tf.nn.l2_loss(inv_s2 - next_states)
                
                elif self.fourier_type == 'dft': 

                    y_target_re = tf.stop_gradient(O - self.coef * tf.matmul(tf.expand_dims(self.con_re, axis=-1), tf.expand_dims(Hth_states, axis=1)) + \
                        (tf.matmul(self.Gamma_re, next_predicted_re) - tf.matmul(self.Gamma_im, next_predicted_im))*(1 - dones))
                    y_re = predicted_re
                    pred_re_loss = self.loss(y_target_re, y_re, target_model)

                    y_target_im = tf.stop_gradient(self.coef * tf.matmul(tf.expand_dims(self.con_re, axis=-1), tf.expand_dims(Hth_states, axis=1)) + \
                        (tf.matmul(self.Gamma_im, next_predicted_re) + tf.matmul(self.Gamma_re, next_predicted_im))*(1 - dones))
                    y_im = predicted_im
                    pred_im_loss = self.loss(y_target_im, y_im, target_model)
                
                else:

                    raise ValueError("Invalid parameter fourier_type {}".format(self.fourier_type))

                # pred_loss = pred_re_loss + pred_im_loss + inv_loss
                pred_loss = pred_re_loss + pred_im_loss

            feature_grad = tape.gradient(pred_loss, self.trainable_variables)
            self.aux_optimizer.apply_gradients(zip(feature_grad, self.trainable_variables))

            grads_proj = None
            if self.use_projection == True:
                grads_proj = tape.gradient(pred_loss, self.projection.trainable_weights)
            
            grads_pred = tape.gradient(pred_loss, self.prediction.trainable_weights)

            del tape

        return pred_loss, pred_re_loss, pred_im_loss, grads_proj, grads_pred


def calculate_layer_units(state_dim, action_dim, ofe_block, total_units, num_layers):
    assert total_units % num_layers == 0

    if ofe_block == "densenet":
        per_unit = total_units // num_layers
        state_layer_units = [per_unit] * num_layers
        action_layer_units = [per_unit] * num_layers

    elif ofe_block in ["mlp"]:
        state_layer_units = [total_units + state_dim] * num_layers
        action_layer_units = [total_units * 2 + state_dim + action_dim] * num_layers

    elif ofe_block in ["mlp_cat"]:
        state_layer_units = [total_units] * num_layers
        action_layer_units = [total_units * 2] * num_layers

    else:
        raise ValueError("invalid connection type")

    return state_layer_units, action_layer_units



class Projection(tf.keras.Model):
    def __init__(self, classifier_type="mlp", output_dim=256, trainable=True):  # input_shape=[128,17]
        super().__init__(name="Projection")
        self.classifier_type = classifier_type
        self.output_dim = output_dim
        self.dense1 = layers.Dense(output_dim*2,
                                      trainable=trainable,
                                      name="projection1")
        self.dense2 = layers.Dense(output_dim,
                                      trainable=trainable,
                                      name="projection2")
        self.BatchNorm1d = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.flatten = layers.Flatten(name="flatten1")

        self.conv1 = layers.Conv2D(
            32, 4, 2, activation='relu', data_format="channels_last", name="proj_conv1"
        )
        # self.poolmax1 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="proj_poolmax1")
        self.conv2 = layers.Conv2D(
            64, 2, 1, activation='relu', data_format="channels_last", name="proj_conv2"
        )
        self.poolmax2 = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="proj_poolmax2")
        
    def call(self, inputs, training=True):

        # inputs = tf.expand_dims(inputs, axis=-1)
        # x = self.conv1(inputs)
        # # x = self.poolmax1(x)
        # x = self.conv2(x)
        # x = self.poolmax2(x)
        # x = self.flatten(x)

        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.BatchNorm1d(x, training=training)  # training=True: The layer will normalize its inputs using the mean and variance of the current batch of inputs.
        x = self.relu(x)
        # x = x * tf.keras.activations.sigmoid(x)
        x = self.dense2(x)

        return x


class Projection2(tf.keras.Model):
    def __init__(self, classifier_type="mlp", output_dim=256, trainable=True):  # input_shape=[128,17]
        super().__init__(name="Projection")
        self.classifier_type = classifier_type
        self.output_dim = output_dim
        self.dense1 = layers.Dense(output_dim*2,
                                      trainable=trainable,
                                      name="projection1")
        self.dense2 = layers.Dense(output_dim,
                                      trainable=trainable,
                                      name="projection2")
        self.BatchNorm1d = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        
    def call(self, inputs, training=True):

        x = self.dense1(inputs)
        x = self.BatchNorm1d(x, training=training)
        x = self.relu(x)
        # x = x * tf.keras.activations.sigmoid(x)
        x = self.dense2(x)

        return x


# class Projection(tf.keras.Model):
#     def __init__(self, classifier_type="mlp", output_dim=256, trainable=True):  # input_shape=[128,17]
#         super().__init__(name="Projection")
#         self.classifier_type = classifier_type
#         self.output_dim = output_dim
#         self.dense1 = layers.Dense(output_dim*2,
#                                       trainable=trainable,
#                                       name="projection1", input_shape=(2176,))
#         self.dense2 = layers.Dense(output_dim,
#                                       trainable=trainable,
#                                       name="projection2", input_shape=(512,))
#         self.BatchNorm1d = layers.BatchNormalization()
#         self.relu = layers.Activation('relu')
#         self.flatten = layers.Flatten(name="flatten1", input_shape=[128,17])

#     def call(self, inputs, if_record=False, index=0, training = True):

#         # if self.classifier_type == "mlp":
#         x = self.flatten(inputs)
#         x1 = self.dense1(x)
#         x2 = self.BatchNorm1d(x1, training = training)  # training=True: The layer will normalize its inputs using the mean and variance of the current batch of inputs.
#         x3 = self.relu(x2)
#         x4 = self.dense2(x3)

#         img_save_dir = './results/img/periodicity'
#         env_name = 'HalfCheetah-v2'
#         dir_of_env = {'HalfCheetah-v2': 'hc', 'HalfCheetah_noise-v2': 'hcNoise', 'HalfCheetah_transfer-v2': 'hcTrans', 
#             'Hopper-v2': 'hopper',  'HopperP-v2': 'hopperP', 'HopperV-v2': 'hopperV', 'Hopper_noise-v2': 'hopperNoise', 
#             'Walker2d-v2': 'walker', 'Walker2dP-v2': 'walkerP',  'Walker2dV-v2': 'walkerV', 'Walker2d_noise-v2': 'walkerNoise', 
#             'Ant-v2': 'ant', 'AntP-v2': 'antP', 'AntV-v2': 'antV', 'Ant_noise-v2': 'antNoise', 
#             'Swimmer-v2': 'swim', 
#             'Humanoid-v2': 'human'}
#         if if_record == True:
#             legend = ['fourier_pred_re_proj', 'fourier_pred_im_proj', 'fourier_true_re_proj', 'fourier_true_im_proj']
#             files = ['inputs.csv', 'flatten_output.csv', 'projection1_output.csv', 'projection1_bn_output.csv', 'projection1_relu_output.csv', 'projection2_output.csv']
#             files_png = ['inputs', 'flatten_output', 'projection1_output', 'projection1_bn_output', 'projection1_relu_output', 'projection2_output']
#             lens = [inputs.shape[1], x.shape[-1], x1.shape[-1], x2.shape[-1], x3.shape[-1], x4.shape[-1]]
#             i = index
            
#             filename = os.path.join(img_save_dir, dir_of_env[env_name], 'inputs.csv')
#             if i==0:
#                 data = pd.DataFrame()
#                 data.to_csv(filename)
#             df = pd.read_csv(filename)
#             df[legend[i]] = tf.squeeze(inputs)[:, 0]
#             df.to_csv(filename, index=None)
            
#             filename = os.path.join(img_save_dir, dir_of_env[env_name], 'flatten_output.csv')
#             if i==0:
#                 data = pd.DataFrame()
#                 data.to_csv(filename)
#             df = pd.read_csv(filename)
#             df[legend[i]] = tf.squeeze(x)
#             df.to_csv(filename, index=None)
            
#             filename = os.path.join(img_save_dir, dir_of_env[env_name], 'projection1_output.csv')
#             if i==0:
#                 data = pd.DataFrame()
#                 data.to_csv(filename)
#             df = pd.read_csv(filename)
#             df[legend[i]] = tf.squeeze(x1)
#             df.to_csv(filename, index=None)
            
#             filename = os.path.join(img_save_dir, dir_of_env[env_name], 'projection1_bn_output.csv')
#             if i==0:
#                 data = pd.DataFrame()
#                 data.to_csv(filename)
#             df = pd.read_csv(filename)
#             df[legend[i]] = tf.squeeze(x2)
#             df.to_csv(filename, index=None)

#             filename = os.path.join(img_save_dir, dir_of_env[env_name], 'projection1_relu_output.csv')
#             if i==0:
#                 data = pd.DataFrame()
#                 data.to_csv(filename)
#             df = pd.read_csv(filename)
#             df[legend[i]] = tf.squeeze(x3)
#             df.to_csv(filename, index=None)

#             filename = os.path.join(img_save_dir, dir_of_env[env_name], 'projection2_output.csv')
#             if i==0:
#                 data = pd.DataFrame()
#                 data.to_csv(filename)
#             df = pd.read_csv(filename)
#             df[legend[i]] = tf.squeeze(x4)
#             df.to_csv(filename, index=None)

#             if i==1:
#                 for j in range(len(files)):
#                     df = pd.read_csv(os.path.join(img_save_dir, dir_of_env[env_name], files[j]))
#                     fig, ax = plt.subplots()
#                     x = np.arange(lens[j])
#                     for k in range(0,2):
#                         ax.plot(x, df.iloc[:, k+1])
#                     ax.legend(legend[0:2])

#                     ax.set_xlabel('x')
#                     ax.set_ylabel('value')
#                     ax.set_title('the projection layer_{}_pred'.format(files_png[j]))

#                     plt.savefig(os.path.join(img_save_dir, dir_of_env[env_name], 'layer_{}_pred'.format(files_png[j])), dpi=300)
#                     plt.close ('all')  
            
#             if i==3:
#                 for j in range(len(files)):
#                     df = pd.read_csv(os.path.join(img_save_dir, dir_of_env[env_name], files[j]))
#                     fig, ax = plt.subplots()
#                     x = np.arange(lens[j])
#                     for k in range(2,4):
#                         ax.plot(x, df.iloc[:, k+1])
#                     ax.legend(legend[2:4])

#                     ax.set_xlabel('x')
#                     ax.set_ylabel('value')
#                     ax.set_title('the projection layer_{}_true'.format(files_png[j]))

#                     plt.savefig(os.path.join(img_save_dir, dir_of_env[env_name], 'layer_{}_true'.format(files_png[j])), dpi=300)
#                     plt.close ('all')  

#         return x4


class QL1Head(tf.keras.Model):
    def __init__(self, head, dueling=False, type="noisy advantage"):
        super().__init__()
        self.head = head
        self.noisy = "noisy" in type
        self.dueling = dueling
        self.encoders = nn.ModuleList()
        self.relu = "relu" in type
        value = "value" in type
        advantage = "advantage" in type
        if self.dueling:
            if value:
                self.encoders.append(self.head.value[1])
            if advantage:
                self.encoders.append(self.head.advantage_hidden[1])
        else:
            self.encoders.append(self.head.network[1])

        self.out_features = sum([e.out_features for e in self.encoders])

    def call(self, x):
        x = x.flatten(-3, -1)
        representations = []
        for encoder in self.encoders:
            encoder.noise_override = self.noisy
            representations.append(encoder(x))
            encoder.noise_override = None
        representation = torch.cat(representations, -1)
        if self.relu:
            representation = F.relu(representation)

        return representation


class Prediction(tf.keras.Model):
    def __init__(self, dim_discretize, dim_state, trainable=True):
        super().__init__(name="Prediction")
        self.output_dim = dim_discretize * dim_state

        self.pred_layer = layers.Dense(1024, # 256
                                      trainable=trainable,
                                      name="pred0") 
        self.out_layer_re = layers.Dense(self.output_dim,
                                      trainable=trainable,
                                      name="pred1_out_re")    
        self.out_layer_im = layers.Dense(self.output_dim,
                                      trainable=trainable,
                                      name="pred1_out_im")
        self.BatchNorm1d = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.flatten = layers.Flatten()

    def call(self, inputs, training=True):

        x = self.pred_layer(inputs)
        x = self.BatchNorm1d(x, training=training)  # training=True: The layer will normalize its inputs using the mean and variance of the current batch of inputs.
        x = self.relu(x)
        # x  = x * tf.keras.activations.sigmoid(x)
        return self.out_layer_re(x), self.out_layer_im(x)