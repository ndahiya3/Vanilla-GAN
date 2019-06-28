#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:29:14 2019
Implementation of Vanilla GAN.

Ref: https://medium.com/ai-society/gans-from-scratch-1-a-deep-introduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
Ref: https://github.com/diegoalejogm/gans
Ref: https://arxiv.org/abs/1406.2661 [Ian Goodfellow]

@author: ndahiya
"""
import torch
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
  
def mnist_data():
  compose = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
       ]
      )
  out_dir = './dataset'
  return datasets.MNIST(root=out_dir, train=True, transform=compose,
                        download=True)
  
def images_to_vectors(images):
  return images.view(images.size(0), 784)

def vectors_to_images(vectors):
  return vectors.view(vectors.size(0), 1, 28, 28)

def noise(size):
  """
  Generates a 1-d vector of gaussian sampled random values
  """
  n = Variable(torch.randn(size, 100))
  return n

def ones_target(size):
  """
  Tensor containing ones, with shape = size
  """
  data = Variable(torch.ones(size, 1))
  return data

def zeros_target(size):
  """
  Tensor containing zeros, with shape = size
  """
  data = Variable(torch.zeros(size, 1))
  return data

