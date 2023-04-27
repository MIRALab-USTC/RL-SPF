import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# Implementation of Online Feature Extractor Network (OFENet)
# Paper: https://arxiv.org/abs/2003.01629

def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain("relu")
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


# Returns an action for a given state
class Actor(nn.Module):
	def __init__(self, state_features_dim, action_dim, max_action, hidden_dim=256):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_features_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)
		self.max_action = max_action
		self.apply(weight_init)

	def forward(self, state_features):
		a = F.relu(self.l1(state_features))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


# Returns Q-value for given state/action pairs
class Critic(nn.Module):
	def __init__(self, state_action_features_dim, hidden_dim=256):
		super(Critic, self).__init__()
		# Q1 architecture
		self.l1 = nn.Linear(state_action_features_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_action_features_dim, hidden_dim)
		self.l5 = nn.Linear(hidden_dim, hidden_dim)
		self.l6 = nn.Linear(hidden_dim, 1)
		self.apply(weight_init)

	def forward(self, hidden_state_action):
		sa = hidden_state_action

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	def Q1(self, hidden_state_action):
		sa = hidden_state_action
		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3_OFE(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		device,
		feature_extractor,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		hidden_dim=256,
	):
		self._extractor = feature_extractor
		self.device = device

		self.actor = Actor(self._extractor.state_features_dim, action_dim, max_action, hidden_dim).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(self._extractor.state_action_features_dim, hidden_dim).to(self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0

	def select_action(self, state):
		state = torch.as_tensor(state.reshape(1, -1), device=self.device, dtype=torch.float32)
		state_features = self._extractor.features_of_states(state)
		action = self.actor(state_features)
		return action.cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample batches from replay buffer  
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		with torch.no_grad():
			_next_state_features = self._extractor.features_of_states(next_state)  # compute Z(o_{t+1})
			# Select action according to policy and add clipped noise
			noise = (torch.randn_like(action) * self.policy_noise).clamp(min=-self.noise_clip, max=self.noise_clip)
			_next_action = (self.actor_target(_next_state_features) + noise).clamp(min=-self.max_action, max=self.max_action)
			sa_hidden_target_state = self._extractor.features_of_states_actions(next_state, _next_action)  # compute Z(o_{t+1}, a_{t+1})

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(sa_hidden_target_state)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		sa_hidden_state = self._extractor.features_of_states_actions(state, action)
		current_Q1, current_Q2 = self.critic(sa_hidden_state)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		actor_loss = torch.tensor(0)
		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:
			# Compute actor loss
			_state_features = self._extractor.features_of_states(state)  # Z(o_t)
			policy_action = self.actor(_state_features)
			sa_policy_hidden_state = self._extractor.features_of_states_actions(state, policy_action) 
			actor_loss = -self.critic.Q1(sa_policy_hidden_state).mean()
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return actor_loss.item(), critic_loss.item()

	# save the model
	def save(self, filename):
		torch.save(self.critic.state_dict(), os.path.join(filename, "_critic"))
		torch.save(self.critic_optimizer.state_dict(), os.path.join(filename, "_critic_optimizer"))
		
		torch.save(self.actor.state_dict(), os.path.join(filename, "_actor"))
		torch.save(self.actor_optimizer.state_dict(), os.path.join(filename, "_actor_optimizer"))

	# load the model
	def load(self, filename):
		self.critic.load_state_dict(torch.load(os.path.join(filename, "_critic")))
		self.critic_optimizer.load_state_dict(torch.load(os.path.join(filename, "_critic_optimizer")))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(os.path.join(filename, "_actor")))
		self.actor_optimizer.load_state_dict(torch.load(os.path.join(filename, "_actor_optimizer")))
		self.actor_target = copy.deepcopy(self.actor)