"""
policy gradient using pytorch
gym, pygame needs to be installed
"""
# 참고 자료: https://github.com/simoninithomas/Policy_gradients_CartPole/blob/master/Siraj's%20Challenge%20Policy%20Gradient%20Learning.ipynb
import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical
import numpy as np
import torch.optim as optim

EPISODES = 500
MAX_STEPS = 100
STATE_SIZE = 4
ACTION_SIZE = 2
GAMMA = 0.95
EPS = np.finfo(np.float32).eps.item()
LR = 5e-3
RENDER = False


class policyNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(STATE_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, ACTION_SIZE),
            nn.Softmax()
        )

    def forward(self, state):
        return self.layers(state)


class Agent():
    def __init__(self, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.history_log_probs = []
        self.history_rewards = []
        self.network = policyNET()
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.network(state) # get action probability using network
        action_category = Categorical(probs)
        action = action_category.sample() # select action idx
        self.history_log_probs.append(action_category.log_prob(action)) # add natural log of selected action probability

        return action.item()


    def train_network(self):
        R = 0
        policy_loss = []
        accumulated_reward = []

        for r in self.history_rewards[::-1]:
            R = r + self.gamma * R
            accumulated_reward.insert(0, R)
        accumulated_reward = torch.tensor(accumulated_reward)
        scaled_accum_reward = (accumulated_reward - accumulated_reward.mean()) / (accumulated_reward.std() + EPS)
        for log_prob, R in zip(self.history_log_probs, scaled_accum_reward):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.history_log_probs = []
        self.history_rewards = []



env = gym.make('CartPole-v0')
running_reward = 10
agent = Agent(LR, GAMMA)


for episode in range(EPISODES+1):
    state = env.reset()
    episode_reward = 0
    for step in range(1, MAX_STEPS+1):
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        if RENDER:
            env.render()
        agent.history_rewards.append(reward)
        episode_reward += reward
        if done:
            break

    running_reward = (1- GAMMA) * episode_reward + GAMMA * running_reward
    agent.train_network()
    if episode % 10 == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
              episode, episode_reward, running_reward))
    if running_reward > env.spec.reward_threshold:
        print("Solved! Running reward is now {} and "
              "the last episode runs to {} time steps!".format(running_reward, step))
        break













'''

for i_episode in range(20):
    observation = env.reset()
    # observation = [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    for t in range(5):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

'''



