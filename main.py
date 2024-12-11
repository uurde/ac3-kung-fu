#importing the libraries
import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import ale_py
import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import ObservationWrapper

#Creating the architecture of the Neural Network
class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 4,  out_channels = 32, kernel_size = (3,3), stride = 2)
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
        self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3,3), stride = 2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2a = torch.nn.Linear(128, action_size)
        self.fc2s = torch.nn.Linear(128, 1)
    
    def forward(self, state):
        x = self.conv1(state)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)[0]
        return action_values, state_value

#setting up environment
class PreprocessAtari(ObservationWrapper):
  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self):
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset()
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    self.frames = self.observation(obs)

def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("State shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.env.env.env.get_action_meanings())

learning_rate = 10e-4
discount_factor = 0.99
number_envrionments = 10

#implement the a3c class
class Agent():
    def __init__(self, action_size):
       self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       self.action_size = action_size
       self.network = Network(action_size).to(self.device)
       self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)

    def act(self, state):
        if state.ndim == 3:
            state = [state]
        state = torch.tensor(state, dtype = torch.float32, device = self.device)
        action_values, _ = self.network(state)
        policy = F.softmax(action_values, dim = -1)
        return np.array([np.random.choice(len(p), p = p) for p in policy.detach().cpu().numpy()])

    def step(self, state, action, reward, next_state, done):
        batch_size = state.shape[0]
        state = torch.tensor(state, dtype = torch.float32, device = self.device)
        next_state = torch.tensor(next_state, dtype = torch.float32, device = self.device)
        reward = torch.tensor(reward, dtype = torch.float32, device = self.device)
        done = torch.tensor(done, dtype = torch.bool, device = self.device).to(dtype = torch.float32)
        action_values, state_value = self.network(state)
        _, next_state_value = self.network(next_state)
        target_state_value = reward + discount_factor * next_state_value * (1 - done)
        advantage = target_state_value - state_value
        probs = F.softmax(action_values, dim = -1)
        logprobs = F.log_softmax(action_values, dim = -1)
        entropy = -torch.sum(probs * logprobs, axis = -1)
        batch_idx = np.arange(batch_size)
        logp_actions = logprobs[batch_idx, action]
        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        critic_loss = F.mse_loss(target_state_value.detach(), state_value)
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

#initializing the a3c agent
agent = Agent(number_actions)

#evaluating our A3C agent on a certain number of episodes
def evaluate(agent, env, n_episodes = 1):
   episodes_rewards = []
   for _ in range(n_episodes):
    state, _ = env.reset()
    total_reward = 0
    while True:
       action = agent.act(state)
       state, reward, done, info, _ = env.step(action[0])
       total_reward += reward
       if done:
          break
    episodes_rewards.append(total_reward)
    return episodes_rewards