#https://github.com/awill139/d3qn_pytorch
import torch
import numpy as np
from torch import optim
import torch.nn as nn

import gym
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size

        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done

        self.mem_cntr += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class DuelingDeepQNet(nn.Module):
    def __init__(self, n_actions, input_dim, fc1_dims, fc2_dims, lr=0.0003):
        super(DuelingDeepQNet, self).__init__()

        self.fc1 = nn.Linear(*input_dim, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.V = nn.Linear(fc2_dims, 1)
        self.A = nn.Linear(fc2_dims, n_actions)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.optim = optim.Adam(self.parameters(), lr=lr)
        self.crit = nn.MSELoss()

    def forward(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))

        V = self.V(x)
        A = self.A(x)

        Q = V + (A - torch.mean(A, dim=1, keepdim=True))

        return Q

    def advantage(self, state):
        x = self.relu1(self.fc1(state))
        x = self.relu2(self.fc2(x))

        return self.A(x)


class Agent:
    def __init__(self, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_decay=1e-8, eps_min=0.01,
                 mem_size=1000000, fc1_dims=128, fc2_dims=128, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.eps_min = eps_min
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(max_size=mem_size, input_shape=input_dims)
        self.q_eval = DuelingDeepQNet(n_actions=n_actions, input_dim=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.q_next = DuelingDeepQNet(n_actions=n_actions, input_dim=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims)

        self.q_eval.to(device)
        self.q_next.to(device)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            state = torch.Tensor([observation]).to(device)
            advantage = self.q_eval.advantage(state)
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

        states, actions, rewards, states_, dones = self.memory.sample(self.batch_size)

        states = torch.tensor(states).to(device)
        rewards = torch.tensor(rewards).to(device)
        dones = torch.tensor(dones).to(device)
        actions = torch.tensor(actions).to(device)
        states_ = torch.tensor(states_).to(device)

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval(states)[indices, actions]
        q_next = self.q_next(states_)

        max_actions = torch.argmax(self.q_eval(states_), dim=1)
        # q_eval = self.q_eval(torch.Tensor(states_).to(device))[indices, actions]
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        q_next[dones] = 0.0
        self.q_eval.optim.zero_grad()

        loss = self.q_eval.crit(q_target, q_pred)
        loss.backward()

        self.q_eval.optim.step()

        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.eps_min else self.eps_min
        self.learn_step_counter += 1


def main():
    env = gym.make('LunarLander-v2')
    # env_id = "CartPole-v0"
    # env = gym.make(env_id)
    n_actions = env.action_space.n
    obs_shape = list(env.observation_space.shape)
    agent = Agent(gamma=0.99, n_actions=n_actions, epsilon=1.0, batch_size=64, input_dims=obs_shape)

    n_games = 1000
    scores = []
    eps_history = []



    for i in range(n_games):
        done = False
        score = 0
        obs = env.reset()


        while not done:
            env.render()
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, obs_, int(done))
            score += reward
            obs = obs_
            agent.learn()

        eps_history.append(agent.epsilon)

        scores.append(score)


        avg_score = np.mean(scores[-100:])

        print('episode: {}\t curr_score: {}\t avg score: {}'.format(i, score, avg_score))


    env.close()

if __name__ == '__main__':
    main()