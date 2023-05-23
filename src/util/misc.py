# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import gin
import numpy as np
import tensorflow as tf


@gin.configurable
def swish(features):
    with tf.name_scope("swish"):
        return features * tf.nn.sigmoid(features)

@gin.configurable
def tanh(features):
    with tf.name_scope("tanh"):
        return tf.nn.tanh(features)

def set_gpu_device_growth():
    configuration = tf.compat.v1.ConfigProto()
    configuration.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=configuration)
    print('available gpu:', tf.config.experimental.list_physical_devices("GPU"))
    for cur_device in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(cur_device, enable=True)
        print("Physical GPUs:", cur_device)


def get_target_dim(env_name):
    TARGET_DIM_DICT = {
        "Ant-v2": 27,
        "HalfCheetah-v2": 17,
        "Walker2d-v2": 17,
        "Hopper-v2": 11,
        "Reacher-v2": 11,
        "Humanoid-v2": 292,
        "Swimmer-v2": 8,
        "InvertedDoublePendulum-v2": 11
    }

    return TARGET_DIM_DICT[env_name]

def get_default_steps(env_name):
    if env_name.startswith('HalfCheetah'):
        default_steps = 3000000
    elif env_name.startswith('Hopper'):
        default_steps = 1000000
    elif env_name.startswith('Walker2d'):
        default_steps = 5000000
    elif env_name.startswith('Ant'):
        default_steps = 5000000
    elif env_name.startswith('Swimmer'):
        default_steps = 3000000
    elif env_name.startswith('Humanoid'):
        default_steps = 3000000
    elif env_name.startswith('InvertedDoublePendulum'):
        default_steps = 1000000

    return default_steps


def make_ofe_name(ofe_layer, ofe_unit, ofe_act, ofe_block):
    exp_name = "L{}_U{}_{}_{}".format(ofe_layer, ofe_unit, ofe_act, ofe_block)
    return exp_name


class EmpiricalNormalization:
    """Normalize mean and variance of values based on emprical values.
    Args:
        shape (int or tuple of int): Shape of input values except batch axis.
        batch_axis (int): Batch axis.
        eps (float): Small value for stability.
        dtype (dtype): Dtype of input values.
        until (int or None): If this arg is specified, the link learns input
            values until the sum of batch sizes exceeds it.
    """

    def __init__(self, shape, batch_axis=0, eps=1e-2, dtype=np.float32,
                 until=None, clip_threshold=None):
        dtype = np.dtype(dtype)
        self.batch_axis = batch_axis
        self.eps = dtype.type(eps)
        self.until = until
        self.clip_threshold = clip_threshold
        self._mean = np.expand_dims(np.zeros(shape, dtype=dtype), batch_axis)
        self._var = np.expand_dims(np.ones(shape, dtype=dtype), batch_axis)
        self.count = 0

        # cache
        self._cached_std_inverse = None

    @property
    def mean(self):
        return np.squeeze(self._mean, self.batch_axis).copy()

    @property
    def std(self):
        return np.sqrt(np.squeeze(self._var, self.batch_axis))

    @property
    def _std_inverse(self):
        if self._cached_std_inverse is None:
            self._cached_std_inverse = (self._var + self.eps) ** -0.5

        return self._cached_std_inverse

    def experience(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[self.batch_axis]
        if count_x == 0:
            return

        self.count += count_x
        rate = x.dtype.type(count_x / self.count)

        mean_x = np.mean(x, axis=self.batch_axis, keepdims=True)
        var_x = np.var(x, axis=self.batch_axis, keepdims=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (
                var_x - self._var
                + delta_mean * (mean_x - self._mean)
        )

        # clear cache
        self._cached_std_inverse = None

    def __call__(self, x, update=True):
        """Normalize mean and variance of values based on emprical values.
        Args:
            x (ndarray or Variable): Input values
            update (bool): Flag to learn the input values
        Returns:
            ndarray or Variable: Normalized output values
        """

        mean = np.broadcast_to(self._mean, x.shape)
        std_inv = np.broadcast_to(self._std_inverse, x.shape)

        if update:
            self.experience(x)

        normalized = (x - mean) * std_inv
        if self.clip_threshold is not None:
            normalized = np.clip(
                normalized, -self.clip_threshold, self.clip_threshold)
        return normalized

    def inverse(self, y):
        mean = np.broadcast_to(self._mean, y.shape)
        std = np.broadcast_to(np.sqrt(self._var + self.eps), y.shape)
        return y * std + mean
