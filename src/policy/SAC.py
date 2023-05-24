import os
import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import trfl.target_update_ops as target_update
from tensorflow.python.training.tracking import tracking

layers = tf.keras.layers


# Actor Network
class GaussianActor(tf.keras.Model):
    LOG_SIG_CAP_MAX = 2
    LOG_SIG_CAP_MIN = -20
    EPS = 1e-6

    def __init__(self, state_dim, action_dim, max_action, layer_units=(256, 256), name='gaussian_policy'):
        super().__init__(name=name)

        base_layers = []
        for cur_layer_size in layer_units:
            cur_layer = layers.Dense(cur_layer_size)
            base_layers.append(cur_layer)

        self.base_layers = base_layers

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        self.out_sigma = layers.Dense(action_dim, name="L_sigma")

        self._max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        self(dummy_state)

    def _dist_from_states(self, states):
        features = states

        for cur_layer in self.base_layers:
            features = cur_layer(features)
            features = tf.nn.relu(features)

        mu_t = self.out_mean(features)

        log_sigma_t = self.out_sigma(features)
        log_sigma_t = tf.clip_by_value(log_sigma_t, self.LOG_SIG_CAP_MIN, self.LOG_SIG_CAP_MAX)

        dist = tfp.distributions.MultivariateNormalDiag(loc=mu_t, scale_diag=tf.exp(log_sigma_t))

        return dist

    def call(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        actions = tf.tanh(raw_actions)

        diff = tf.reduce_sum(tf.math.log(1 - actions ** 2 + self.EPS), axis=1)
        log_pis -= diff

        actions = actions * self._max_action
        return actions, log_pis

    def mean_action(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.mean()
        actions = tf.tanh(raw_actions) * self._max_action

        return actions


class CriticV(tf.keras.Model):
    def __init__(self, state_dim, layer_units=(256, 256), name='vf'):
        super().__init__(name=name)

        base_layers = []
        for cur_layer_size in layer_units:
            cur_layer = layers.Dense(cur_layer_size)
            base_layers.append(cur_layer)

        self.base_layers = base_layers
        self.out_layer = layers.Dense(1, name="V_out")

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        self(dummy_state)

    def call(self, states):
        features = states

        for cur_layer in self.base_layers:
            features = cur_layer(features)
            features = tf.nn.relu(features)

        values = self.out_layer(features)
        values = tf.squeeze(values, axis=1, name="values")
        return values


class CriticQ(tf.keras.Model):
    def __init__(self, state_action_dim, layer_units=(256, 256), name='vq'):
        super().__init__(name=name)

        base_layers = []
        for cur_layer_size in layer_units:
            cur_layer = layers.Dense(cur_layer_size)
            base_layers.append(cur_layer)

        self.base_layers = base_layers
        self.out_layer = layers.Dense(1, name="Q_out")

        dummy_state = tf.constant(np.zeros(shape=[1, state_action_dim], dtype=np.float32))
        self(dummy_state)

    def call(self, inputs):
        features = inputs

        for cur_layer in self.base_layers:
            features = cur_layer(features)
            features = tf.nn.relu(features)

        values = self.out_layer(features)
        values = tf.squeeze(values, axis=1)
        return values


def get_default_scale_reward(env_name):
    if env_name.startswith("Ant"):
        return 5.0
    elif env_name.startswith("Walker2d"):
        return 5.0
    elif env_name.startswith("HalfCheetah"):
        return 5.0
    elif env_name.startswith("Hopper"):
        return 5.0
    elif env_name.startswith("Swimmer"):
        return 5.0
    elif env_name.startswith("InvertedDoublePendulum"):
        return 5.0
    elif env_name.startswith("Humanoid"):
        return 20.0
    else:
        raise ValueError("received an invalid environment name.")


class InferenceProxy(tracking.AutoTrackable):
    def __init__(self, actor, ofe_net=None):
        self.actor = actor

        if ofe_net is not None:
            self.ofe_net = ofe_net.get_inference_proxy()
        else:
            self.ofe_net = None

    @tf.function(autograph=False, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def inference(self, states):
        if self.ofe_net is not None:
            states = self.ofe_net.features_from_states(states)

        actions = self.actor.mean_action(states)

        return actions


# @gin.configurable(blacklist=["state_dim", "action_dim", "max_action"])
# @gin.configurable
class SAC(tracking.AutoTrackable):

    def __init__(self, state_dim, action_dim, max_action, scale_reward,
                 feature_extractor,
                 learning_rate=3e-4,
                 actor_units=(256, 256),
                 q_units=(256, 256),
                 v_units=(256, 256),
                 tau=0.005):
        self.scale_reward = scale_reward
        self.actor = GaussianActor(feature_extractor.dim_state_features, action_dim, max_action,
                                   layer_units=actor_units)
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        self.vf = CriticV(feature_extractor.dim_state_features, layer_units=v_units)
        self.vf_target = CriticV(feature_extractor.dim_state_features, layer_units=v_units)
        self.vf_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        target_update.update_target_variables(self.vf_target.weights, self.vf.weights)
        self.qf1 = CriticQ(feature_extractor.dim_state_action_features, layer_units=q_units, name="vq1")
        self.qf2 = CriticQ(feature_extractor.dim_state_action_features, layer_units=q_units, name="vq2")

        self.qf1_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.qf2_optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        # Constants
        self.tau = tau

        self.ofe_net = feature_extractor
        self._inference_proxy = None

    def select_action(self, state):
        """

        :param state:
        :return:
        """
        state = tf.cast(state, tf.float32)

        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)     
            action = self._select_action_body(state)
            action = tf.squeeze(action, axis=0)
        else:
            action = self._select_action_body(state)

        return action.numpy()

    @tf.function(autograph=False)
    def _select_action_body(self, state):
        """

        :param np.ndarray state:
        :return:
        """
        state_feature = self.ofe_net.features_from_states(state)
        action = self.actor.mean_action(state_feature)
        return action

    def select_action_noise(self, state):
        """selects action with noise for exploration.

        :param state:
        :return:
        """
        state = tf.cast(state, tf.float32)

        if len(state.shape) == 1:
            state = tf.expand_dims(state, axis=0)     
            action = self._select_action_noise_body(state)
            action = tf.squeeze(action, axis=0)
        else:
            action = self._select_action_noise_body(state)

        return action.numpy()

    @tf.function(autograph=False)
    def _select_action_noise_body(self, state):
        """

        :param state:
        :return:
        """
        state_feature = self.ofe_net.features_from_states(state)
        action, _ = self.actor(state_feature)
        return action

    @tf.function(autograph=False)
    def train_for_batch(self, states, actions, rewards, next_states, done, discount=0.99):
        """

        :param states:
        :param actions:
        :param rewards:
        :param next_states:
        :param done:
        :param discount:
        :return:
        """
        assert len(done.shape) == 2
        assert len(rewards.shape) == 2
        done = tf.squeeze(done, axis=1)
        rewards = tf.squeeze(rewards, axis=1)

        not_done = 1 - tf.cast(done, dtype=tf.float32)

        # Critic Update
        with tf.GradientTape(persistent=True) as tape:
            state_action_features = self.ofe_net.features_from_states_actions(states, actions)
            next_state_features = self.ofe_net.features_from_states(next_states)

            q1 = self.qf1(state_action_features)
            q2 = self.qf2(state_action_features)
            vf_next_target_t = self.vf_target(next_state_features)

            # Equation (7, 8)
            ys = tf.stop_gradient(
                self.scale_reward * rewards + not_done * discount * vf_next_target_t
            )

            td_loss1 = tf.reduce_mean((ys - q1) ** 2)
            td_loss2 = tf.reduce_mean((ys - q2) ** 2)

        # Equation (9)
        q1_grad = tape.gradient(td_loss1, self.qf1.trainable_variables)
        self.qf1_optimizer.apply_gradients(zip(q1_grad, self.qf1.trainable_variables))
        q2_grad = tape.gradient(td_loss2, self.qf2.trainable_variables)
        self.qf2_optimizer.apply_gradients(zip(q2_grad, self.qf2.trainable_variables))

        del tape

        # Actor Update
        with tf.GradientTape(persistent=True) as tape:
            state_features = self.ofe_net.features_from_states(states)

            vf_t = self.vf(state_features)
            sample_actions, log_pi = self.actor(state_features)

            state_action_features = self.ofe_net.features_from_states_actions(states, sample_actions)

            q1 = self.qf1(state_action_features)
            q2 = self.qf2(state_action_features)
            min_q = tf.minimum(q1, q2)

            # Equation (12)
            policy_loss = tf.reduce_mean(log_pi - q1)

            # Equation (5)
            target_v = tf.stop_gradient(min_q - log_pi)
            vf_loss_t = tf.reduce_mean((target_v - vf_t) ** 2)

        # Equation (6)
        vf_grad = tape.gradient(vf_loss_t, self.vf.trainable_variables)
        self.vf_optimizer.apply_gradients(zip(vf_grad, self.vf.trainable_variables))

        # Equation (13)
        actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        del tape

        return policy_loss, vf_loss_t, td_loss1

    def train(self, replay_buffer, batch_size=256, discount=0.99):
        # Sample replay replay_buffer
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)

        with tf.device("/gpu:0"):
            policy_loss, vf_loss, td_loss = self.train_for_batch(states, actions, rewards, next_states, dones,
                                                                 discount=discount)

            target_update.update_target_variables(self.vf_target.weights, self.vf.weights, tau=self.tau)

        tf.summary.scalar(name="sac/V_Loss", data=policy_loss)
        tf.summary.scalar(name="sac/TD_Loss", data=vf_loss)
        tf.summary.scalar(name="sac/ActorLoss", data=policy_loss)

    def save(self, save_dir):
        self.actor.save_weights(os.path.join(save_dir,'agent_actor_model'))
        self.vf.save_weights(os.path.join(save_dir,'agent_vf_model'))
        self.vf_target.save_weights(os.path.join(save_dir,'agent_vf_target_model'))
        self.qf1.save_weights(os.path.join(save_dir,'agent_qf1_model'))
        self.qf2.save_weights(os.path.join(save_dir,'agent_qf2_model'))

    def load(self, load_dir):
        self.actor.load_weights(os.path.join(load_dir,'agent_actor_model'))
        self.vf.load_weights(os.path.join(load_dir,'agent_vf_model'))
        self.vf_target.load_weights(os.path.join(load_dir,'agent_vf_target_model'))
        self.qf1.load_weights(os.path.join(load_dir,'agent_qf1_model'))
        self.qf2.load_weights(os.path.join(load_dir,'agent_qf2_model'))
