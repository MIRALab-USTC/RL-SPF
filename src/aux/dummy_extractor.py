# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

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
