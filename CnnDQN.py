"""
Created on  Feb 27 2021
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
from wrappers import make_atari, wrap_deepmind, wrap_pytorch

USE_CUDA = torch.cuda.is_available()
#将变量放到cuda上
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class CnnDQN(nn.Module):
    def __init__(self, observation_space, action_sapce):
        super(CnnDQN, self).__init__()

        self.observation_space = observation_space
        self.action_sapce = action_sapce

        self.features = nn.Sequential(
            nn.Conv2d(self.observation_space[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512,self.action_sapce)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), -1)#将多维度的Tensor展平成一维
        # x.size(0)指batchsize的值,x = x.view(x.size(0), -1)简化x = x.view(batchsize, -1),view()函数的功能根reshape类似，用来转换size大小。
        # x = x.view(batchsize, -1)中batchsize指转换后有几行，而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数。
        x = self.fc(x)
        return x

    # def feature_size(self):
    #     #这里就很粗暴，先建立一个大小和预期输入的全0tensor，送入features中运行，最后得到输出，展平，读取长度。这里是7 * 7 * 64
    #     return self.features(autograd.Variable(torch.zeros(1, *self.observation_space))).view(1, -1).size(1)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True)#(1,1,84,84)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
            action = action.cpu().numpy()  # 从网络中得到的tensor形式，因为之后要输入给gym环境中，这里把它放回cpu，转为数组形式
            action = int(action)

        else:
            action = random.randrange(self.action_sapce)
        return action

class ReplayBuffer(object):
    def __init__(self, capacity):
        #deque模块是python标准库collections中的一项，它提供了两端都可以操作的序列，其实就是双向队列，
        #可以从左右两端增加元素，或者是删除元素。如果设置了最大长度，非输入端的数据会逐步移出窗口。
        self.buffer = deque (maxlen = capacity)

    def push (self, state ,aciton, reward, next_state, done):
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
    env_id = "PongNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    observation_space = env.observation_space.shape
    action_sapce = env.action_space.n

    model = CnnDQN(observation_space, action_sapce)

    if USE_CUDA:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters())

    replay_buffer = ReplayBuffer(1000)

    batch_size = 32
    gamma = 0.99
    replay_initial = 100
    num_frames = 14000

    losses = []
    all_rewards = []
    x_axis1 = []
    x_axis2= []
    episode_reward = 0

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000

    # 要求探索率随着迭代次数增加而减小
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    state = env.reset()

    for frame_idx in range(1, num_frames + 1):
        #显示动画
        env.render()
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

        if frame_idx+1 > replay_initial:
            loss = compute_td_loss(model, optimizer, replay_buffer, gamma, batch_size)
            x_axis2.append(frame_idx)
            losses.append(np.array(loss.data.cpu()))



        if frame_idx % 100 == 0:
            plt.figure(1)
            plt.subplot(121)
            plt.plot(x_axis1, all_rewards)
            plt.subplot(122)
            plt.plot(x_axis2, losses)
            plt.show()

    env.close()








if __name__ == '__main__':
    main()