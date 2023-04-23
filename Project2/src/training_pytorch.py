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

    # check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load data
    (x_train, y_train), _ = get_dataset(args.dataset)

    # convert data to PyTorch tensors and create DataLoader
    data_tensor = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
    labels_tensor = torch.tensor(y_train, dtype=torch.long)
    train_set = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    # load ResNet model
    if args.resnet_size == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
    elif args.resnet_size == 'resnet101':
        model = torchvision.models.resnet101(pretrained=False)
    elif args.resnet_size == 'resnet152':
        model = torchvision.models.resnet152(pretrained=False)
    else:
        print('Unsupported ResNet model')
        quit()

    # Change the output layer to have 10 classes instead of 1000
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(np.unique(y_train)))

    # send model to GPU
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate=0.001, momentum=0.9)

    # Train the model
    for epoch in range(10):  # number of epochs
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs and labels
            inputs, labels = data

            # send data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

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
    parser.add_argument('--resnet_size', type=str, default='resnet50',
                        help='Size for Resnet, one in ["resnet50", "resnet101", "resnet152"]')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train the NN, if none it will based on pre-defined values')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Path where to save trianing data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)