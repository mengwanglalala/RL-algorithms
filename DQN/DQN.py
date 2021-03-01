"""
Created on  Feb 26 2021
@author: wangmeng
"""

import math, random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from collections import  Counter
from collections import deque
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
#将变量放到cuda上
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class DQN(nn.Module):
    def __init__(self, observation_space, action_sapce):
        super(DQN, self).__init__()

        self.observation_space = observation_space
        self.action_sapce = action_sapce

        self.layers = nn.Sequential(
            nn.Linear(observation_space,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128, action_sapce)
        )


    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            #如果使用的是GPU，这里需要把数据丢到GPU上
            state   = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True)#volatile的作用是作为指令关键字，确保本条指令不会因编译器的优化而省略，且要求每次直接读值。
            #.squeeze() 把数据条目中维度为1 的删除掉
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
            #max(1)返回每一行中最大值的那个元素，且返回其索引,max(0)是列
            #max()[1]只返回最大值的每个索引，max()[0]， 只返回最大值的每个数

            action = action.cpu().numpy()#从网络中得到的tensor形式，因为之后要输入给gym环境中，这里把它放回cpu，转为数组形式
            action =int(action)
        else:
            action = random.randrange(self.action_sapce)#返回指定递增基数集合中的一个随机数，基数默认值为1。
        return action

class ReplayBuffer(object):
    def __init__(self, capacity):
        #deque模块是python标准库collections中的一项，它提供了两端都可以操作的序列，其实就是双向队列，
        #可以从左右两端增加元素，或者是删除元素。如果设置了最大长度，非输入端的数据会逐步移出窗口。
        self.buffer = deque(maxlen = capacity)

    def push (self, state, aciton, reward, next_state, done):
        state = np.expand_dims(state,0)
        #这里增加维度的操作是为了便于之后使用concatenate进行拼接
        next_state = np.expand_dims(next_state,0)
        self.buffer.append((state, aciton, reward, next_state, done))

    def sample(self, batch_size):
        # 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        state , action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        #最后使用concatenate对数组进行拼接，相当于少了一个维度
        return np.concatenate(state),  action, reward, np.concatenate(next_state), done


def compute_td_loss(model,optimizer, replay_buffer, gamma, batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    #通通丢到GPU上去
    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))
    done = Variable(torch.FloatTensor(done))

    q_values = model(state)
    next_q_values = model(next_state)

    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    #gather可以看作是对q_values的查询，即元素都是q_values中的元素，查询索引都存在action中。输出大小与action.unsqueeze(1)一致。
    #dim=1,它存放的都是第1维度的索引；dim=0，它存放的都是第0维度的索引；
    #这里增加维度主要是为了方便gather操作，之后再删除该维度
    next_q_value = next_q_values.max(1)[0]

    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss




def main():
    env_id = "CartPole-v0"
    env = gym.make(env_id)

    observation_space = env.observation_space.shape[0]
    action_sapce = env.action_space.n

    model = DQN (observation_space, action_sapce)

    if USE_CUDA:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters())

    replay_buffer = ReplayBuffer(1000)

    batch_size = 32
    gamma = 0.99

    num_frames = 10000

    losses = []
    all_rewards = []
    x_axis1 = []
    x_axis2 = []
    episode_reward = 0

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 500

    #要求探索率随着迭代次数增加而减小
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp( -1. * frame_idx / epsilon_decay)

    state = env.reset()
    for frame_idx in range(1, num_frames + 1):
        #显示动画
        #env.render()
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            x_axis1.append(frame_idx)
            all_rewards.append(episode_reward)
            episode_reward = 0

        if frame_idx+1 > batch_size:
            x_axis2.append(frame_idx)
            loss = compute_td_loss(model, optimizer, replay_buffer, gamma, batch_size)
            losses.append(np.array(loss.data.cpu()))



        if frame_idx % 200 == 0:
            plt.figure(1)
            plt.subplot(121)
            plt.plot(x_axis1, all_rewards)
            plt.subplot(122)
            plt.plot(x_axis2, losses)
            plt.show()


if __name__ == '__main__':
    main()