import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class Features:
    """ 计算特征 """

    def __init__(self, n_bandits=100, decay=0.97, lower_p=0.2, upper_p=0.8):
        # 老虎机个数
        self.n_bandits = n_bandits
        # 衰减系数
        self.decay = decay
        # 步数
        self.step = 0  # 步数(特征1)

        # 每台老虎机的出糖概率
        self.ratings = np.linspace(0.0, 1.0, n_bandits + 1)
        self.dist = np.full((n_bandits, n_bandits + 1), 1.0 / (n_bandits + 1))
        self.total_decay = np.full(n_bandits, 1.0)
        self.expected = np.full(n_bandits, 0.5)  # 出糖期望(特征2)

        # 置信区间
        self.lower_p = lower_p
        self.upper_p = upper_p
        self.pro_lower_bound = np.full(n_bandits, lower_p)  # 置信区间下界(特征3)
        self.pro_upper_bound = np.full(n_bandits, upper_p)  # 置信区间上界(特征4)
        self.interval_length = self.pro_upper_bound - self.pro_lower_bound  # 置信区间长度(特征5)

        # 访问次数
        self.num_selection = np.full(n_bandits, 0)  # 玩家访问次数(特征6)
        self.num_opp_selection = np.full(n_bandits, 0)  # 对手访问次数(特征7)
        self.sum_num = np.full(n_bandits, 0)  # 访问总次数(特征8)
        self.num_ratio = np.full(n_bandits, 0)  # 玩家访问次数比例(特征9)

        # 出糖次数占比
        self.num_reward = np.full(n_bandits, 0)
        self.ratio_reward = np.full(n_bandits, 0.5)  # 得糖占比(特征10)  ？

        # 最近三次得糖情况
        self.get_reward_dis = np.full((n_bandits, 3), 0.5)
        self.get_reward_th = np.full(n_bandits, 0.5)  # (特征11)

        # 轨迹特征
        self.num_opp_before = np.full(n_bandits, 0)  # 对手在玩家之前探索了多少次(特征12)
        self.num_before = np.full(n_bandits, 0)  # 玩家在对手之前探索了多少次(特征13)
        self.num_opp_after = np.full(n_bandits, 0)  # 对手在玩家之后探索了多少次(特征14)
        self.num_after = np.full(n_bandits, 0)  # 玩家在对手之后探索了多少次(特征15)

        self.opp_last_step = np.full(n_bandits, 0)
        self.opp_bunching = np.full(n_bandits, 0.0)  # (特征16)
        self.opp_rep_trace = np.full(n_bandits, 0.0)  # (特征17)
        self.opp_acc_trace = np.full(n_bandits, 0.0)

    def calc_expected(self, action):
        """ 通过机器的概率分布计算出出糖概率期望 """
        return (self.dist[action] * self.ratings * self.total_decay[action]).sum()

    def update_bound(self, action, p):
        """ 更新置信区间上下界 """
        cum_prob = np.cumsum(self.dist[action])
        index = int(np.argmax(cum_prob > p))
        if index == 0:
            return 0.
        upper = cum_prob[index]
        lower = cum_prob[index - 1]
        ratio = (p - lower) / (upper - lower)
        low_exp = (index - 1) / self.n_bandits * self.total_decay[action]
        upp_exp = index / self.n_bandits * self.total_decay[action]
        return ratio * upp_exp + (1 - ratio) * low_exp

    def update_interval_length(self, action):
        """ 更新置信区间长度 """
        self.pro_lower_bound[action] = self.update_bound(action, self.lower_p)
        self.pro_upper_bound[action] = self.update_bound(action, self.upper_p)
        self.interval_length[action] = self.pro_upper_bound[action] - self.pro_lower_bound[action]

    def update_opp_bunching(self, opp_action):
        """ 更新opp_bunching """
        num_opp = self.num_opp_selection[opp_action]
        if num_opp > 1:
            bunch = 1.0 / np.sqrt(self.step - self.opp_last_step[opp_action])
            self.opp_bunching[opp_action] += (bunch - self.opp_bunching[opp_action]) / (num_opp - 1)
        self.opp_last_step[opp_action] = self.step

    def update_opp_trace(self, opp_action, action):
        """ 更新 """
        rep_forget = 0.9
        rep_erase = 0.7
        acc_forget = 0.95
        acc_erase = 0.2
        self.opp_rep_trace[:] *= rep_forget
        self.opp_rep_trace[action] *= rep_erase

        self.opp_acc_trace[:] *= acc_forget
        self.opp_acc_trace[action] *= acc_erase

        if self.num_opp_selection[opp_action] < 2:
            return

        self.opp_rep_trace[opp_action] = 1
        self.opp_acc_trace[opp_action] += 1

    def feature_update(self, step, action, reward, opp_action):
        """ 根据每一步更新特征的值 """
        # 更新步数
        self.step = step
        # 更新概率分布
        priors = self.dist[action, :]
        likelihood = self.ratings * self.total_decay[action]
        if reward == 0:
            likelihood = 1 - likelihood
        self.dist[action, :] = likelihood * priors
        self.dist[action, :] = self.dist[action, :] / self.dist[action, :].sum()
        # 更新总衰减
        self.total_decay[action] *= self.decay
        self.total_decay[opp_action] *= self.decay

        # 更新概率期望
        self.expected[action] = self.calc_expected(action)
        self.expected[opp_action] = self.calc_expected(opp_action)

        # 更新置信区间
        self.update_interval_length(action)
        self.update_interval_length(opp_action)

        # 更新访问次数
        self.num_selection[action] += 1
        self.sum_num[action] += 1
        self.num_opp_selection[opp_action] += 1
        self.sum_num[opp_action] += 1
        self.num_ratio[action] = self.num_selection[action] / self.sum_num[action]

        self.get_reward_dis[action, 0] = self.get_reward_dis[action, 1]
        self.get_reward_dis[action, 1] = self.get_reward_dis[action, 2]
        self.get_reward_dis[action, 2] = 0
        if reward > 0:
            self.num_reward[action] += 1
            self.get_reward_dis[action, 2] = 1
        self.ratio_reward[action] = self.num_reward[action] / self.num_selection[action]
        self.get_reward_th[action] = self.get_reward_dis[action, :].mean()

        if self.num_selection[opp_action] == 0:
            self.num_opp_before[opp_action] += 1
        if self.num_opp_selection[action] == 0:
            self.num_before[action] += 1
        self.num_after[action] += 1
        self.num_opp_after[opp_action] += 1
        self.num_after[opp_action] = 0
        self.num_opp_after[action] = 0

        self.update_opp_bunching(opp_action)
        self.update_opp_trace(opp_action, action)


class ClasssfyNN(nn.Module):
    """ 定义神经网络 """

    def __init__(self, input_size, hidden_size1=16, hidden_size2=16):
        super(ClasssfyNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        # out = self.relu1(out)
        out = nn.functional.tanh(out)
        out = self.fc2(out)
        # out = self.relu2(out)
        out = nn.functional.tanh(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out


class ReplayBuffer:
    """ 定义经验回放缓冲区 """

    def __init__(self, capacity=4000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def refresh(self):
        self.buffer = []
        self.position = 0


class NetModel:
    def __init__(self, input_size):
        self.input_size = input_size
        self.network = ClasssfyNN(input_size)

    def get_value(self, x):
        """ 输入x应为矩阵形式 """
        inputs = torch.from_numpy(x.astype(np.float32))
        outputs = self.network(inputs)
        y = outputs.detach().numpy().T
        return y

    def train(self, data, lr=0.001, num_epochs=100):
        """ 输入训练数据为ReplayBuffer中的buffer列表 """
        # 处理训练数据
        data_train = np.array(data)
        x_train = torch.from_numpy(data_train[:, 0:self.input_size].astype(np.float32))
        y_train = torch.from_numpy(data_train[:, self.input_size:(self.input_size + 1)].astype(np.float32))

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # 训练模型
        for epoch in range(num_epochs):
            # 前向传播
            outputs = self.network(x_train)
            loss = criterion(outputs, y_train)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
