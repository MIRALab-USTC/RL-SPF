# coding=utf-8
# Copyright 2018 The Gin-Config Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified by an anonymous author to be able to run on tf 2.0
# The original code comes from https://github.com/google/gin-config

"""Contains TensorFlow or Google-specific utilities for Gin configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from gin import config
# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.framework import summary_pb2


# pylint: enable=g-direct-tensorflow-import


def write_gin_to_summary(dir_out, global_step):
    """Writes out Gin's operative config, and adds a summary of it.

    :param dir_out:
    :param int global_step:
    :return:
    """
    config_str = config.operative_config_str()

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    filename = 'operative-%s.gin' % (global_step,)
    config_path = os.path.join(dir_out, filename)

    with open(config_path, "w") as f:
        f.write(config_str)

    md_config_str = _markdownify_operative_config_str(config_str)
    summary_metadata = summary_pb2.SummaryMetadata()
    summary_metadata.plugin_data.plugin_name = 'text'
    summary_metadata.plugin_data.content = b'{}'

    tf.summary.write(tag="gin", tensor=md_config_str, metadata=summary_metadata, step=global_step)
    tf.summary.flush()


def _markdownify_operative_config_str(string):
    """Convert an operative config string to markdown format."""

    # TODO: Total hack below. Implement more principled formatting.
    def process(line):
        """Convert a single line to markdown format."""
        if not line.startswith('#'):
            return '    ' + line

        line = line[2:]
        if line.startswith('===='):
            return ''
        if line.startswith('None'):
            return '    # None.'
        if line.endswith(':'):
            return '#### ' + line
        return line

    output_lines = []
    for line in string.splitlines():
        procd_line = process(line)
        if procd_line is not None:
            output_lines.append(procd_line)

    return '\n'.join(output_lines)
