
"""
TD3 implementation in Tensorflow Eager Execution

This implementation follows the setting in
 https://github.com/sfujim/TD3/blob/master/TD3.py

batch size is 256 instead of 100, which is written in the above paper.

"""

import os
import gin
import numpy as np
import tensorflow as tf
import trfl.target_update_ops as target_update
from tensorflow.python.training.tracking import tracking

layers = tf.keras.layers
regularizers = tf.keras.regularizers
losses = tf.keras.losses


class Actor(tf.keras.Model):
    def __init__(self, state_feature_dim, action_dim, max_action, layer_units=(400, 300), name="Actor"):
        super().__init__(name=name)

        initializer = tf.keras.initializers.Orthogonal()
        base_layers = []
        for cur_layer in layer_units:
            base_layers.append(layers.Dense(cur_layer, kernel_initializer=initializer))

        self.base_layers = base_layers
        self.out_layer = layers.Dense(action_dim, kernel_initializer=initializer)

        self.max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=[1, state_feature_dim], dtype=np.float32))
        self(dummy_state)

    @tf.function
    def call(self, inputs):
        features = inputs

        with tf.device("/gpu:0"):
            for cur_layer in self.base_layers:
                features = cur_layer(features)
                features = tf.nn.relu(features)

            features = self.out_layer(features)
            action = self.max_action * tf.nn.tanh(features)

        return action


class Critic(tf.keras.Model):
    def __init__(self, state_action_feature_dim, layer_units=(400, 300), name="Critic"):
        super().__init__(name=name)

        initializer = tf.keras.initializers.Orthogonal()
        base_layers = []
        for cur_layer in layer_units:
            base_layers.append(layers.Dense(cur_layer, kernel_initializer=initializer))

        self.base_layers1 = base_layers

        base_layers = []
        for cur_layer in layer_units:
            base_layers.append(layers.Dense(cur_layer, kernel_initializer=initializer))

        self.base_layers2 = base_layers

        self.out1 = layers.Dense(1, name="out1", kernel_initializer=initializer)
        self.out2 = layers.Dense(1, name="out2", kernel_initializer=initializer)

        dummy_features = tf.constant(np.zeros(shape=[1, state_action_feature_dim], dtype=np.float32))
        self(dummy_features)

    @tf.function
    def call(self, inputs):
        with tf.device("/gpu:0"):
            features = inputs

            for cur_layer in self.base_layers1:
                features = cur_layer(features)
                features = tf.nn.relu(features)

            features1 = self.out1(features)

            features = inputs
            for cur_layer in self.base_layers2:
                features = cur_layer(features)
                features = tf.nn.relu(features)

            features2 = self.out2(features)

        return features1, features2

    @tf.function
    def Q1(self, inputs):
        with tf.device("/gpu:0"):
            features = inputs

            for cur_layer in self.base_layers1:
                features = cur_layer(features)
                features = tf.nn.relu(features)

            features = self.out1(features)

        return features


# @gin.configurable(blacklist=["state_dim", "action_dim", "max_action"])
class TD3(tracking.AutoTrackable):
    def __init__(self, state_dim, action_dim, max_action, feature_extractor, linear_range=1e5, layer_units=(400, 300), action_noise=0.1):
        self._extractor = feature_extractor
        self._action_noise = action_noise

        self.actor = Actor(self._extractor.dim_state_features, action_dim, max_action, layer_units=layer_units)
        self.actor_target = Actor(self._extractor.dim_state_features, action_dim, max_action, layer_units=layer_units)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=3e-4)

        # initialize target network
        for param, target_param in zip(self.actor.weights, self.actor_target.weights):
            target_param.assign(param)

        self.critic = Critic(self._extractor.dim_state_action_features, layer_units=layer_units)
        self.critic_target = Critic(self._extractor.dim_state_action_features, layer_units=layer_units)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=3e-4)

        # initialize target network
        for param, target_param in zip(self.critic.weights, self.critic_target.weights):
            target_param.assign(param)

        self._max_action = max_action

        self._update_count = 0

    def select_action(self, state):
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
        features = self._extractor.features_from_states(states)
        action = self.actor(features)
        return action
    
    def select_action_noise(self, state):
        assert isinstance(state, np.ndarray)
        state =  state.astype(np.float32)
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
            action = self._select_action_body(tf.constant(state))
            # action = tf.clip_by_value(action + tf.random.normal(shape=action.shape, stddev=self._action_noise), \
            #                       -self._max_action, self._max_action)  # modified
            action = tf.clip_by_value(
                action + tf.random.normal(shape=action.shape, stddev=self._action_noise*self._max_action), \
                -self._max_action, self._max_action)
            action = tf.squeeze(action, axis=0)
        else:
            action = self._select_action_body(state)  # modified
            action = tf.clip_by_value(action + tf.random.normal(shape=action.shape, stddev=self._action_noise), \
                                  -self._max_action, self._max_action)  # modified

        return action.numpy()

    @tf.function
    def _select_action_target_body(self, states):  # modified
        features = self._extractor.features_from_states(states)
        action = self.actor_target(features)
        return action

    def train(self, replay_buffer, batch_size=256, discount=0.99, tau=0.005):
        self._update_count += 1

        state, action, next_state, reward, done = replay_buffer.sample(batch_size)
        done = np.array(done, dtype=np.float32)
        critic_loss = self._critic_update(state, action, next_state, reward, done, discount)

        tf.summary.scalar(name="loss/CriticLoss", data=critic_loss)

        if self._update_count % 2 == 0:
            actor_loss = self._actor_update(state, tau)
            tf.summary.scalar(name="loss/ActorLoss", data=actor_loss)
        else:
            actor_loss = None
        tf.summary.scalar(name='loss/action_noise', data=self._action_noise)

        return actor_loss, critic_loss

    @tf.function
    def _critic_update(self, states, actions, next_states, rewards, done, discount, policy_noise=0.2, noise_clip=0.5):
        with tf.device("/gpu:0"):
            not_done = 1 - done

            next_states_features = self._extractor.features_from_states(next_states)
            next_action = self.actor_target(next_states_features)
            noise_shape = tf.shape(next_action)
            action_noise = tf.random.normal(noise_shape, mean=0.0, stddev=policy_noise*self._max_action) # modify
            noise_clip = noise_clip * self._max_action
            action_noise = tf.clip_by_value(action_noise, -noise_clip, noise_clip)

            next_actions = tf.clip_by_value(next_action + action_noise, -self._max_action, self._max_action)

            with tf.GradientTape() as tape:
                next_sa_features = self._extractor.features_from_states_actions(next_states, next_actions)
                target_Q1, target_Q2 = self.critic_target(next_sa_features)
                # target_Q = rewards + (not_done * discount * target_Q)
                target_Q = rewards + (not_done * discount * tf.minimum(target_Q1, target_Q2))
                # detach => stop_gradient
                target_Q = tf.stop_gradient(target_Q)

                sa_features = self._extractor.features_from_states_actions(states, actions)
                current_Q1, current_Q2 = self.critic(sa_features)

                # Compute critic loss
                critic_loss = tf.reduce_mean(losses.MSE(current_Q1, target_Q)) + \
                              tf.reduce_mean(losses.MSE(current_Q2, target_Q))

            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        return critic_loss

    @tf.function
    def _actor_update(self, states, tau):

        with tf.device("/gpu:0"):
            with tf.GradientTape() as tape:
                states_features = self._extractor.features_from_states(states)
                policy_actions = self.actor(states_features)
                sa_features = self._extractor.features_from_states_actions(states, policy_actions)
                actor_loss = -tf.reduce_mean(self.critic.Q1(sa_features))

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            # Update target networks
            target_update.update_target_variables(self.critic_target.weights, self.critic.weights, tau)
            target_update.update_target_variables(self.actor_target.weights, self.actor.weights, tau)

        return actor_loss

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