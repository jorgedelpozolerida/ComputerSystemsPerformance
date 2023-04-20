#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training pytorch Resnet


"""

import os
import sys
import argparse
from utils import get_dataset

import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'datain')

def main(args):
    (x_train, y_train), _ = get_dataset(args.dataset)
    x_train = torch.from_numpy(x_train)
    y_train = np.array(y_train)
    print(y_train, len(x_train))

    train_set = torch.utils.data.TensorDataset(x_train, y_train)

    

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    print("Trainloader",trainloader)

    # Define the ResNet50 model
    model = torchvision.models.resnet50(pretrained=False)

    # Change the output layer to have 10 classes instead of 1000
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):  # number of epochs
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs and labels
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to train NN on, one in ["MNIST", "CIFAR10", "ImageNet"]')
    parser.add_argument('--resnet_size', type=int, default=None,
                        help='Size for Resnet, one in [50, 101, 152]')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train the NN, if none it will based on pre-defined values')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Path where to save trianing data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)