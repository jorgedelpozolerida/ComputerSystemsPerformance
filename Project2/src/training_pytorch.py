#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training pytorch Resnets


"""

import os
import io
import sys
import argparse
from utils import get_dataset, get_modeloutputdata, write_to_file
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'datain')
EXPORT_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)))

def main(args):
    file_name  = f"run{args.run}-{args.device}-epoch{args.epochs}-batchsize{args.batch_size}-pytorch-{args.dataset}-{args.resnet_size}_MODEL.csv"
    csv_path = os.path.join(EXPORT_PATH, 'experiments', args.device, 'pytorch', file_name)

    print(csv_path)
    write_to_file("epoch;step;loss_value;timestamp", csv_path)

    # Print the parsed arguments
    print(f"Framework: Pytorch")
    print(f"Dataset: {args.dataset}")
    print(f"Resnet size: {args.resnet_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    # print(f"Output directory: {args.out_dir}")
    print("")

    # Get all prints before training away
    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    
    # Handle device used
    assert args.device in ['cpu', 'gpu'], "Please select only 'cpu' or 'gpu' as device"
    device_name = args.device
    if args.device == 'gpu':
        # check if GPU is available
        if torch.cuda.is_available():
            device_name = "cuda:0" 
            print("GPU: ",device_name)
        else:
            raise ValueError("There is no gpu available, please use cpu")
        
    print(device_name)
    device = torch.device(device_name)

    # Load data
    (x_train, y_train), (x_test, y_test) = get_dataset(args.dataset)

    # Convert data to PyTorch tensors and create DataLoader
    batch_size = int(args.batch_size)
    data_tensor = torch.from_numpy(x_train).permute(0, 3, 1, 2).float()
    labels_tensor = torch.tensor(y_train, dtype=torch.long)
    train_set = torch.utils.data.TensorDataset(data_tensor, labels_tensor)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load ResNet model
    if args.resnet_size == 'resnet50':
        model = torchvision.models.resnet50(weights=None)
    elif args.resnet_size == 'resnet101':
        model = torchvision.models.resnet101(weights=None)
    elif args.resnet_size == 'resnet152':
        model = torchvision.models.resnet152(weights=None)
    else:
        print('Unsupported ResNet model')
        quit()

    # Change the output layer to have 10 classes instead of 1000
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(np.unique(y_train)))

    # Send model to GPU(CPU)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # Reset stdout to its original value
    sys.stdout = sys.__stdout__

    # Train the model
    n_epochs = int(args.epochs)

    # create header for csv file
    write_to_file("epoch;precision;recall;accuracy;timestamp", csv_path)
    
    for epoch in range(n_epochs):  # number of epochs
        running_loss = 0.0
        for step, data in enumerate(trainloader, 0):
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
            if step % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        percision = precision_score(y_test, y_pred)

        # in the end of each epoch write model eval metrics
        write_to_file(get_modeloutputdata([epoch,percision,recall,acc]), csv_path)

    print('Finished Training')


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to train NN on, one in ["SVHN", "CIFAR10", "CIFAR100"]')
    parser.add_argument('--resnet_size', type=str, default='resnet50',
                        help='Size for Resnet, one in ["resnet50", "resnet101", "resnet152"]')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train the NN, if not provided set to default')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size, if not provided set to default')
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device: cpu or gpu')
    parser.add_argument('--run', type=int, default=0,
                        help='# of experiment')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)