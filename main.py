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
        super(nn.Module, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = (3, 3), stride = 2)
        self.conv2 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = 2)
        self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), stride = 2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512, 128)
        self.fc2a = torch.nn.Linear(128, action_size)
        self.fc2s = torch.nn.Linear(128, 1)

print("test")