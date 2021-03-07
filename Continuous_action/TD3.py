
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class GaussianExploration(object):
    def __init__(self, action_space, max_sigma=1.0, min_sigma=1.0, decay_period=1000000):
        self.low = action_space.low
        self.high = action_space.high
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period

    def get_action(self, action, t=0):
        sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action = action + np.random.normal(size=len(action)) * sigma
        return np.clip(action, self.low, self.high)


def soft_update(net, target_net, soft_tau=1e-2):
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]

class TD(object):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(TD, self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 128
        self.gamma = 0.99
        self.soft_tau = 1e-2
        self.noise_std = 0.2
        self.noise_clip = 0.5
        self.policy_update = 2
        self.soft_tau = 1e-2
        self.replay_buffer_size = 1000000
        self.value_lr = 1e-3
        self.policy_lr = 1e-3

        self.value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net1 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_value_net2 = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        soft_update(self.value_net1, self.target_value_net1, soft_tau=1.0)
        soft_update(self.value_net2, self.target_value_net2, soft_tau=1.0)
        soft_update(self.policy_net, self.target_policy_net, soft_tau=1.0)

        self.value_criterion = nn.MSELoss()

        self.value_optimizer1 = optim.Adam(self.value_net1.parameters(), lr=self.value_lr)
        self.value_optimizer2 = optim.Adam(self.value_net2.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)


    def td3_update(self, step, batch_size):

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action     = torch.FloatTensor(action).to(device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        next_action = self.target_policy_net(next_state)
        noise = torch.normal(torch.zeros(next_action.size()), self.noise_std).to(device)
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
        next_action += noise

        target_q_value1  = self.target_value_net1(next_state, next_action)
        target_q_value2  = self.target_value_net2(next_state, next_action)
        target_q_value   = torch.min(target_q_value1, target_q_value2)
        expected_q_value = reward + (1.0 - done) * self.gamma * target_q_value

        q_value1 = self.value_net1(state, action)
        q_value2 = self.value_net2(state, action)

        value_loss1 = self.value_criterion(q_value1, expected_q_value.detach())
        value_loss2 = self.value_criterion(q_value2, expected_q_value.detach())

        self.value_optimizer1.zero_grad()
        value_loss1.backward()
        self.value_optimizer1.step()

        self.value_optimizer2.zero_grad()
        value_loss2.backward()
        self.value_optimizer2.step()

        if step % self.policy_update == 0:
            policy_loss = self.value_net1(state, self.policy_net(state))
            policy_loss = -policy_loss.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            soft_update(self.value_net1, self.target_value_net1, soft_tau=self.soft_tau)
            soft_update(self.value_net2, self.target_value_net2, soft_tau=self.soft_tau)
            soft_update(self.policy_net, self.target_policy_net, soft_tau=self.soft_tau)

def main():
    env = NormalizedActions(gym.make('Pendulum-v0'))
    noise = GaussianExploration(env.action_space)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256

    TD3 = TD(action_dim, state_dim, hidden_dim)


    max_frames = 10000
    max_steps = 500
    frame_idx = 0
    rewards = []
    batch_size = 128

    while frame_idx < max_frames:
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = TD3.policy_net.get_action(state)
            action = noise.get_action(action, step)
            next_state, reward, done, _ = env.step(action)

            TD3.replay_buffer.push(state, action, reward, next_state, done)
            if len(TD3.replay_buffer) > batch_size:
                TD3.td3_update(step, batch_size)

            state = next_state
            episode_reward += reward
            frame_idx += 1

            if frame_idx % 1000 == 0:
                plot(frame_idx, rewards)

            if done:
                break

        rewards.append(episode_reward)

if __name__ == '__main__':
    main()




