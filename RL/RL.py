import random
import warnings
import torch
import torch.nn as nn
import numpy as np
import gym
import time

warnings.filterwarnings("ignore", category=UserWarning)



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 24),  # 输入层到隐藏层的全连接层，输入维度为4，输出维度为24
            nn.ReLU(),  # 隐藏层激活函数使用ReLU
            nn.Linear(24, 24),  # 隐藏层到隐藏层的全连接层，输入维度为24，输出维度为24
            nn.ReLU(),
            nn.Linear(24, 2)  # 隐藏层到输出层的全连接层，输入维度为24，输出维度为2
        )
        self.mse_loss = nn.MSELoss()  # 均方误差损失函数
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # Adam优化器，学习率为0.001

    def forward(self, inputs):
        return self.fc(inputs)


def train_cartpole(num_iterations):
    env = gym.envs.make('CartPole-v1')  # 创建CartPole环境
    env = env.unwrapped  # 取消环境的限制
    net = MyNet()  # 创建一个神经网络
    net2 = MyNet()  # 创建一个用于更新参数的目标网络

    store_count = 0  # 存储的样本数量
    store_size = 2000  # 存储的样本容量
    decline = 0.6  # epsilon的衰减率
    learn_time = 0  # 训练次数
    update_time = 20  # 更新目标网络的频率
    gamma = 0.9  # 折扣因子
    batch_size = 1000  # 每次训练的批次大小
    store = np.zeros((store_size, 10))  # 存储样本的数组
    start_study = False  # 是否开始训练的标志
    longest_duration = 0  # 最长存活时间

    for i in range(num_iterations):
        state = env.reset()  # 重置环境
        start_time = time.time()  # 记录起始时间

        while True:
            if random.randint(0, 100) < 100 * (decline ** learn_time):
                action = random.randint(0, 1)  # 随机选择动作
            else:
                output = net(torch.Tensor(state)).detach()  # 使用网络预测动作值
                action = torch.argmax(output).data.item()  # 选择具有最高动作值的动作

            next_state, reward, done, info = env.step(action)  # 执行动作并观察下一个状态、奖励和完成标志
            reward = (env.theta_threshold_radians - abs(next_state[2])) / env.theta_threshold_radians * 0.7 + \
                     (env.x_threshold - abs(next_state[0])) / env.x_threshold * 0.3  # 对奖励进行调整

            store[store_count % store_size][0:4] = state  # 存储当前状态
            store[store_count % store_size][4:5] = action  # 存储选择的动作
            store[store_count % store_size][5:9] = next_state  # 存储下一个状态
            store[store_count % store_size][9:10] = reward  # 存储奖励
            store_count += 1
            state = next_state

            if store_count > store_size:
                if learn_time % update_time == 0:
                    net2.load_state_dict(net.state_dict())  # 更新目标网络的参数

                index = random.randint(0, store_size - batch_size - 1)  # 随机选择样本
                batch_states = torch.Tensor(store[index:index + batch_size, 0:4])  # 批量状态
                batch_actions = torch.Tensor(store[index:index + batch_size, 4:5]).long()  # 批量动作

                batch_next_states = torch.Tensor(store[index:index + batch_size, 5:9])  # 批量下一个状态
                batch_rewards = torch.Tensor(store[index:index + batch_size, 9:10])  # 批量奖励

                q_values = net(batch_states).gather(1, batch_actions)  # 计算当前状态下选择的动作的值
                next_q_values = net2(batch_next_states).detach().max(1)[0].reshape(batch_size, 1)  # 计算下一个状态的最大动作值
                target_q_values = batch_rewards + gamma * next_q_values  # 计算目标Q值
                loss = net.mse_loss(q_values, target_q_values)  # 计算损失函数

                net.optimizer.zero_grad()  # 清空梯度
                loss.backward()  # 反向传播计算梯度
                net.optimizer.step()  # 更新网络参数

                learn_time += 1
                if not start_study:
                    print('start to study')
                    start_study = True
                    break

            if done:
                end_time = time.time()  # 记录结束时间
                duration = end_time - start_time  # 计算持续时间
                print(f"Iteration: {i+1}, Duration: {duration:.2f} seconds")
                if duration > longest_duration:
                    longest_duration = duration  # 更新最长存活时间
                break

            env.render()  # 渲染环境

    print(f"Longest Duration: {longest_duration:.2f} seconds")


if __name__ == '__main__':
    train_cartpole(1000)  # 运行训练函数
