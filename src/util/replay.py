# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.

import numpy as np
import scipy.signal
from src.util.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, distribute_value
import os

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
        """

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

    def save(self, save_dir):
        np.savez(os.path.join(save_dir,'rb_data'), states=self.states, 
                                                next_states=self.next_states, 
                                                actions=self.actions, 
                                                rewards=self.rewards, 
                                                dones=self.dones)
        return

    def load(self, load_dir):
        data = np.load(os.path.join(load_dir,'rb_data.npz'))
        self.states = data['states']
        self.next_states = data['next_states']
        self.actions = data['actions']
        self.rewards = data['rewards']
        self.dones = data['dones']
        return


class PPOBuffer:
    """A buffer for storing trajectories experienced by a PPO agent.
    Uses Generalized Advantage Estimation (GAE-Lambda) for calculating
    the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def add(self, obs, act, obs2, rew, done, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size    # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.obs2_buf[self.ptr] = obs2
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.

        input: 
            vector x, 
            [x0, 
            x1, 
            x2]

        output:
            [x0 + discount * x1 + discount^2 * x2,  
            x1 + discount * x2,
            x2]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def finish_path(self, last_val=0):
        """
        use to compute_returns_and_advantages

        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1] # r + gamma*v(s') - v(s)
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

        return

    def get(self):
        """
        Returns data stored in buffer.
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        # self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # advantage normalization trick
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-5)

        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]

    def sample(self, batch_size=100):
        ind = np.random.randint(0, self.ptr, size=batch_size)

        cur_states = self.obs_buf[ind, :]
        cur_next_states = self.obs2_buf[ind, :]
        cur_actions = self.act_buf[ind, :]
        cur_rewards = self.rew_buf[ind]
        cur_dones = self.done_buf[ind]

        return cur_states, cur_actions, cur_next_states, cur_rewards.reshape(-1, 1), cur_dones.reshape(-1, 1)