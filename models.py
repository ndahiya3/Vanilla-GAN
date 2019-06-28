#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:41:37 2019
Implementation of Vanilla GAN.

Ref: https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
Ref: https://github.com/diegoalejogm/gans
Ref: https://arxiv.org/abs/1406.2661 [Ian Goodfellow]
@author: ndahiya
"""

import torch
from torch import nn

class DiscriminatorNet(torch.nn.Module):
  """
  A three hidden layer Discriminator Network.
  """
  def __init__(self):
    super(DiscriminatorNet, self).__init__()
    n_features = 784 # 28x28
    n_out = 1
    
    self.hidden0 = nn.Sequential(
        nn.Linear(n_features, 1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
        )
    self.hidden1 = nn.Sequential(
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
        )
    self.hidden2 = nn.Sequential(
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3)
        )
    self.out = nn.Sequential(
        torch.nn.Linear(256, n_out),
        torch.nn.Sigmoid()
        )
  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    
    return x
  
class GeneratorNet(torch.nn.Module):
  """
  A three hidden layer Generator Network.
  Converts a 100 component noise vector to a 784 [28x28] component mnist
  handwritten digit vector.
  """
  def __init__(self):
    super(GeneratorNet, self).__init__()
    n_features = 100 # 100 component noise vector
    n_out = 784
    
    self.hidden0 = nn.Sequential(
        nn.Linear(n_features, 256),
        nn.LeakyReLU(0.2)
        )
    self.hidden1 = nn.Sequential(
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2)
        )
    self.hidden2 = nn.Sequential(
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2)
        )
    self.out = nn.Sequential(
        nn.Linear(1024, n_out),
        nn.Tanh()
        )
  def forward(self, x):
    x = self.hidden0(x)
    x = self.hidden1(x)
    x = self.hidden2(x)
    x = self.out(x)
    
    return x
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

