#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Script with functions to be used across files
"""

import os

from torchvision import datasets, transforms

import numpy as np                                                                  # NOQA E402
import datetime

# constant which denotes where to cache the data
THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'datain')
EXPERIMENTS_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'experiments')



###########################################
# ------------ Data loading ---------------
###########################################

def load_MNIST(save_dir, test_split=0.1):
    '''
    Downloads and saves the MINST data set to the save_dit location and returns a
    split numpy array (by the test_split fraction)
    '''
    
    data_raw = datasets.ImageNet(root=save_dir,split='train',  train=True, download=True, transform=transforms.ToTensor())
    data = data_raw.data.numpy()
    data=np.stack((data, data, data), axis=3)

    labels = data_raw.targets.numpy()
    pivot = int(len(data) * (1-test_split))
    
    data_train, data_test = data[0:pivot] , data[pivot: len(data)]
    labels_train, labels_test = labels[0:pivot] , labels[pivot: len(data)]

    return (data_train, labels_train), (data_test, labels_test)

def load_SVHN(save_dir, test_split=0.1):
    '''
    Downloads and saves the SVHN data set to the save_dir location and returns a
    split numpy array (by the test_split fraction)
    '''
    data_raw = datasets.SVHN(root=save_dir, split='train', download=True, transform=transforms.ToTensor())
    data = data_raw.data.transpose((0, 2, 3, 1)) # Transpose the dimensions to (N, H, W, C)
    labels = data_raw.labels
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
    Downloads and saves the CIFAR100 data set to the save_dir location and returns a
    split numpy array (by the test_split fraction)
    '''
    data_raw = datasets.CIFAR100(root=save_dir, train=True, download=True, transform=transforms.ToTensor())
    data = data_raw.data
    labels = data_raw.targets
    pivot = int(len(data) * (1-test_split))
    
    data_train, data_test = data[0:pivot] , data[pivot: len(data)]
    labels_train, labels_test = labels[0:pivot] , labels[pivot: len(data)]

    return (data_train, labels_train), (data_test, labels_test)


def get_dataset(dataset: str, test_split=0.1, savedir=DATAIN_PATH):
    if dataset == "CIFAR10":
        return load_CIFAR10(savedir, test_split)
    elif dataset == "CIFAR100":
        return load_CIFAR100(savedir, test_split)
    elif dataset == "SVHN":
        return load_SVHN(savedir, test_split) 
    else:
        raise ValueError("Invalid dataset name")
    
    
###########################################
# ------------ Data saving ---------------
###########################################


def get_modeloutputdata(epoch, step, loss_value):
    '''
    Returns a formated row to be added to csv
    '''
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    return f"{epoch};{step};{loss_value};{timestamp}"

def write_to_file(data: str, path: str):
    with open(path, "a+") as file:
        file.write(data+"\n")