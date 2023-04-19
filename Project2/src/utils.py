#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script with functions to be used across files

"""

import os
import sys
import argparse


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402


# TODO:
# - function to load MNIST, CIFAR, IMAGENET
# - function to preprocess each dataset to fit each of the ResNets
# - function that performs the splits  
# - functions to be used in training scripts


# TO CONSIDER:
# - same structure of oflder for each dataset to identify where train and test is, where each dataset is, and where each resnet saves it output

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


torch.manual_seed(1234)
    
    
def ensure_dir(dir):
    ''' Creates dir if not exists '''
    if not os.path.exists(dir):
        os.makedirs(dir)
        _logger.info(f"Created dir: {dir}")  # log an INFO level message


    return dir


def load_MNIST(save_dir):
    
    
    
    
    return





def load_CIFAR(save_dir):
    

        
    # Load CIFAR10 dataset
    train_set = datasets.CIFAR10(root=save_dir, train=True, download=True, transform=transforms.ToTensor())

    # Split into training and validation sets
    train_set, val_set = torch.utils.data.random_split(train_set, [45000, 5000])
        
    
    return


def load_ImageNet(save_dir):
    
    return




def generate_trainingdata(out_path):
    
    load_CIFAR(ensure_dir(os.path.join(out_path, "CIFAR")))
    load_MNIST(ensure_dir(os.path.join(out_path, "MNIST")))
    load_ImageNet(ensure_dir(os.path.join(out_path, "ImageNet")))
