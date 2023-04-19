#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Script with functions to be used across files
"""

import os
import sys
import argparse

from torchvision import datasets, transforms

import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402

# constant which denotes where to cache the data
SAVE_DIR = "../datain"

def load_MNIST(save_dir, test_split=0.1):
    '''
    Downloads and saves the MINST data set to the save_dit location and returns a
    split numpy array (by the test_split fraction)
    '''
    data_raw = datasets.MNIST(root=save_dir, train=True, download=True, transform=transforms.ToTensor())
    data = data_raw.data.numpy()
    labels = data_raw.targets.numpy()
    pivot = int(len(data) * (1-test_split))
    
    data_train, data_test = data[0:pivot] , data[pivot: len(data)]
    labels_train, labels_test = labels[0:pivot] , labels[pivot: len(data)]

    return (data_train, labels_train), (data_test, labels_test)

def load_CIFAR10(save_dir, test_split=0.1):
    '''
    Downloads and saves the CIFAR10 data set to the save_dit location and returns a
    split numpy array (by the test_split fraction)
    '''
    data_raw = datasets.CIFAR10(root=save_dir, train=True, download=True, transform=transforms.ToTensor())
    data = data_raw.data
    labels = data_raw.targets
    pivot = int(len(data) * (1-test_split))
    
    data_train, data_test = data[0:pivot] , data[pivot: len(data)]
    labels_train, labels_test = labels[0:pivot] , labels[pivot: len(data)]

    return (data_train, labels_train), (data_test, labels_test)

def load_CIFAR100(save_dir, test_split=0.1):
    '''
    Downloads and saves the CIFAR100 data set to the save_dit location and returns a
    split numpy array (by the test_split fraction)
    '''
    data_raw = datasets.CIFAR100(root=save_dir, train=True, download=True, transform=transforms.ToTensor())
    data = data_raw.data
    labels = data_raw.targets
    pivot = int(len(data) * (1-test_split))
    
    data_train, data_test = data[0:pivot] , data[pivot: len(data)]
    labels_train, labels_test = labels[0:pivot] , labels[pivot: len(data)]

    return (data_train, labels_train), (data_test, labels_test)

def get_dataset(dataset: str, test_split=0.1):
    if dataset == "MNIST":
        return load_MNIST(SAVE_DIR, test_split)
    elif dataset == "CIFAR10":
        return load_CIFAR10(SAVE_DIR, test_split)
    elif dataset == "CIFAR100":
        return load_CIFAR100(SAVE_DIR, test_split)