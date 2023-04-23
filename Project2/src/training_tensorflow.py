#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training tensorflow Resnets


"""

import os
import sys
import argparse
from utils import get_dataset

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'datain')

# TODO:
# - check that dataset to be used exists
# - train NN

def main(args):

    # Handle device used
    assert args.device in ['cpu', 'gpu'], "Please select only 'cpu' or 'gpu' as device"
    if args.device == 'gpu':
        physical_devices = tf.config.list_physical_devices('GPU') # get number of gpu
        # check if GPU is available
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            raise ValueError("There is no gpu available, please use cpu")

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = get_dataset(args.dataset)

    num_classes = len(np.unique(y_train))
    
    # Define ResNet
    shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])      
    if args.resnet_size == "resnet50":
        resnet = keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=shape, pooling="avg", classes=num_classes
        )
    elif args.resnet_size == "resnet101":
        resnet = keras.applications.ResNet101(
            include_top=False, weights=None, input_shape=shape, pooling="avg", classes=num_classes
        )
    elif args.resnet_size == "resnet152":
        resnet = keras.applications.ResNet152(
            include_top=False, weights=None, input_shape=shape, pooling="avg", classes=num_classes
        )
    else:
        raise ValueError("Invalid ResNet model name")

    # Prepare the training dataset
    batch_size = int(args.batch_size)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        # train_dataset.shuffle(buffer_size=1024)
        train_dataset
        .batch(batch_size)
        .map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    inputs = keras.Input(shape=shape)
    x = resnet(inputs, training=True)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    # Define the loss function and optimizer
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

    # Train the model
    n_epochs = int(args.epochs)
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step % 100 == 0:
                print("Training loss at step {}: {:.4f}".format(step, loss_value))

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
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Path where to save training data')
    parser.add_argument('--device', type=str, default='GPU',
                        help='Device: cpu or gpu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)