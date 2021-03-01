# RL-algorithms
更新一些基础的RL代码
- [离散的动作空间](#离散的动作空间)
  - [DQN](#DQN)
  - [DDQN](#DDQN)
  - [Dueling DQN](#Dueling DQN)

- [连续的动作空间](#连续的动作空间)


## 离散的动作空间
(已更) \<br>
### DQN
可用于入门深度强化学习，使用一个Q Network来估计Q值，从而替换了 Q-table，完成从离散状态空间到连续状态空间的跨越。Q Network 会对每一个离散动作的Q值进行估计，执行的时候选择Q值最高的动作（greedy 策略）。并使用 epslion-greedy 策略进行探索（探索的时候，有很小的概率随机执行动作），来获得各种动作的训练数据

(更新中) \<br>
### DDQN
(Double DQN)更加稳定，因为最优化操作会传播高估误差，所以她同时训练两个Q network并选择较小的Q值用于计算TD-error，降低高估误差。

### Dueling DQN
使用了优势函数 advantage function（A3C也用了）：它只估计state的Q值，不考虑动作，好的策略能将state 导向一个更有优势的局面。然而不是任何时刻 action 都会影响 state的转移，因此Dueling DQN 结合了 优势函数估计的Q值 与 原本DQN对不同动作估计的Q值。DQN算法学习 state 与每个离散动作一一对应的Q值后才能知道学到 state 的Q值，而Dueling DQN 能通过优势函数直接学到state的价值，这使得Dueling DQN在一些action不影响环境的情况下能学比DQN更快

### D3QN
Dueling DQN 与Double DQN相互兼容，一起用效果很好。简单，泛用，没有使用禁忌。任何一个刚入门的人都能独立地在前两种算法的基础上改出D3QN。在论文中使用了D3QN应该引用DuelingDQN 与 DoubleDQN的文章

### Noisy DQN
探索能力稍强。Noisy DQN 把噪声添加到网络的输出层之前值。原本Q值较大的动作在添加噪声后Q值变大的概率也比较大。这种探索比epslion-greedy随机选一个动作去执行更好，至少这种针对性的探索既保证了探索动作多样，也提高了探索效率。

## 连续的动作空间
(已更) \<br>
### DDPG
DDPG（Deep DPG ），可用于入门连续动作空间的DRL算法。DPG 确定策略梯度算法，直接让策略网络输出action，成功在连续动作空间任务上训练出能用的策略，但是它使用 OU-noise 这种有很多超参数的方法去探索环境，训练慢，且不稳定。

### PPO（Proximal PO 近端策略搜索）
训练稳定，调参简单，robust（稳健、耐操）。PPO对TRPO的信任域计算过程进行简化，论文中用的词是 surrogate objective。PPO动作的噪声方差是一个可训练的矢量（与动作矢量相同形状），而不由网络输出，这样做增强了PPO的稳健性 robustness。

(更新中) \<br>
### A3C（Asynchronous Advantage Actor-Critic）
Asynchronous 指开启多个actor 在环境中探索，并异步更新。原本DDPG的Critic 是 Q(s, a)，根据state-action pair 估计Q值，优势函数只使用 state 去估计Q值，这是很好的创新：降低了随机策略梯度算法估计Q值的难度。然而优势函数有明显缺陷：不是任何时刻 action 都会影响 state的转移（详见 Dueling DQN），因此这个算法只适合入门学习「优势函数 advantage function」。如果你看到新论文还在使用A3C，那么你要怀疑其作者RL的水平。此外，A3C算法有离散动作版本，也有连续动作版本。A2C 指的是没有Asynchronous 的版本。

### PPO+GAE（Generalized Advantage Estimation）
训练最稳定，调参最简单，适合高维状态 High-dimensional state，但是环境不能有太多随机因数。GAE会根据经验轨迹 trajectory 生成优势函数估计值，然后让Critic去拟合这个值。在这样的调整下，在随机因素小的环境中，不需要太多 trajectory 即可描述当前的策略。尽管GAE可以用于多种RL算法，但是它与PPO这种On-policy 的相性最好。

### SAC（Soft Actor-Critic with maximum entropy 最大熵）
训练很快，探索能力好，但是很依赖Reward Function，不像PPO那样随便整一个Reward function 也能训练。PPO算法会计算新旧策略的差异（计算两个分布之间的距离），并让这个差异保持在信任域内，且不至于太小。SAC算法不是on-policy算法，不容易计算新旧策略的差异，所以它在优化时最大化策略的熵（动作的方差越大，策略的熵越高）

### TD3
TD3（TDDD，Twin Delay DDPG），擅长调参的人才建议用，因为它影响训练的敏感超参数很多。它从Double DQN那里继承了Twin Critic，用来降低高估误差；它用来和随机策略梯度很像的方法：计算用于更新TD-error的Q值时，给action加上了噪声，用于让Critic拟合更平滑的Q值估计函数。TD3建议 延迟更新目标网络，即多更新几次网络后，再使用 soft update 将网络更新到target network上，我认为这没有多大用，后来的其他算法也不用这个技巧。TD3还建议在计算Q值时，为动作添加一个噪声，用于平滑Critic函数，在确定策略中，TD3这么用很像“随机策略”

