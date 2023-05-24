import tensorflow as tf


class DummyFeatureExtractor(object):
    """A feature extractor which output raw values, so this doesn't process inputs.

    This method has the same interface with OFENet.

    """

    def __init__(self, dim_state, dim_action):
        self._dim_state_features = dim_state
        self._dim_state_action_features = dim_state + dim_action

    @property
    def dim_state_features(self):
        return self._dim_state_features

    @property
    def dim_state_action_features(self):
        return self._dim_state_action_features

    def features_from_states(self, states):
        """extracts features from states.
        :return:
        """
        return states

    def features_from_states_actions(self, states, actions):
        """extracts features from states.
        :return:
        """
        features = tf.concat([states, actions], axis=1)

        return features

    def train(self, states, actions, next_states, rewards, dones):
        return
