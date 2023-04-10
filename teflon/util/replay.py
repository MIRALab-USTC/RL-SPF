# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import numpy as np


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, capacity=10000001):
        # [self.start_idx, self.end_idx)
        self.is_full = False
        self.end_idx = 0
        self.capacity = capacity

        self.states = np.zeros([capacity, state_dim], dtype=np.float32)
        self.next_states = np.zeros([capacity, state_dim], dtype=np.float32)
        self.actions = np.zeros([capacity, action_dim], dtype=np.float32)
        self.rewards = np.zeros([capacity], dtype=np.float32)
        self.dones = np.zeros([capacity], dtype=np.bool)

    @property
    def size(self):
        if self.is_full:
            return self.capacity
        else:
            return self.end_idx

    def add(self, state, action, next_state, reward, done):
        self.states[self.end_idx, :] = state
        self.next_states[self.end_idx, :] = next_state
        self.actions[self.end_idx, :] = action
        self.rewards[self.end_idx] = reward
        self.dones[self.end_idx] = done

        self.end_idx += 1
        if self.end_idx == self.capacity:
            self.end_idx = 0
            self.is_full = True

    def sample(self, batch_size=100):
        if not self.is_full:
            ind = np.random.randint(0, self.end_idx, size=batch_size)
        else:
            ind = np.random.randint(0, self.capacity, size=batch_size)

        cur_states = self.states[ind, :]
        cur_next_states = self.next_states[ind, :]
        cur_actions = self.actions[ind, :]
        cur_rewards = self.rewards[ind]
        cur_dones = self.dones[ind]

        return cur_states, cur_actions, cur_next_states, cur_rewards.reshape(-1, 1), cur_dones.reshape(-1, 1)

    def sample_indices(self, indices):
        """指定されたindexのデータ群を返す

        :param indices:
        :return:
        """
        cur_states = self.states[indices, :]
        cur_next_states = self.next_states[indices, :]
        cur_actions = self.actions[indices, :]
        cur_rewards = self.rewards[indices]
        cur_dones = self.dones[indices]

        return cur_states, cur_actions, cur_next_states, cur_rewards.reshape(-1, 1), cur_dones.reshape(-1, 1)

    def sample_recently(self, batch_size, delay=0):
        if not self.is_full:
            if self.end_idx - batch_size - delay < 0:
                raise ValueError("Replay Buffer has too small samples.")

        start_idx = self.end_idx - batch_size - delay

        if start_idx < 0:
            start_idx = self.capacity + (self.end_idx - batch_size - delay)
            cur_end_idx = self.end_idx - delay

            if cur_end_idx < 0:
                cur_end_idx = self.capacity + (self.end_idx - delay)
                cur_states = self.states[start_idx:cur_end_idx]
                cur_next_states = self.next_states[start_idx:cur_end_idx]
                cur_actions = self.actions[start_idx:cur_end_idx]
                cur_rewards = self.rewards[start_idx:cur_end_idx]
                cur_dones = self.dones[start_idx:cur_end_idx]

            else:
                cur_states = np.concatenate((self.states[start_idx:], self.states[:cur_end_idx]), axis=0)
                cur_next_states = np.concatenate((self.next_states[start_idx:], self.next_states[:cur_end_idx]), axis=0)
                cur_actions = np.concatenate((self.actions[start_idx:], self.actions[:cur_end_idx]), axis=0)
                cur_rewards = np.concatenate((self.rewards[start_idx:], self.rewards[:cur_end_idx]), axis=0)
                cur_dones = np.concatenate((self.dones[start_idx:], self.dones[:cur_end_idx]), axis=0)
        else:
            offset = start_idx
            cur_states = self.states[offset:offset + batch_size, :]
            cur_next_states = self.next_states[offset:offset + batch_size, :]
            cur_actions = self.actions[offset:offset + batch_size, :]
            cur_rewards = self.rewards[offset:offset + batch_size]
            cur_dones = self.dones[offset:offset + batch_size]

        return cur_states, cur_actions, cur_next_states, cur_rewards.reshape(-1, 1), cur_dones.reshape(-1, 1)
