#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" {Short Description of Script}


{Long Description of Script}
"""
__author__ = "Jorge del Pozo Lérida"
__email__ =  "jorgedelpozolerida@gmail.com"
__date__ =  "18/04/2023"



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
    (x_train, y_train), (x_test, y_test) = get_dataset(args.dataset)

    # Prepare the training dataset
    batch_size = 128
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.shuffle(buffer_size=1024)
        .batch(batch_size)
        .map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Define the ResNet50 model
    shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    num_classes = 10
    resnet50 = keras.applications.ResNet50(
        include_top=False, weights=None, input_shape=shape, pooling="avg", classes=num_classes
    )
    inputs = keras.Input(shape=shape)
    x = resnet50(inputs, training=True)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    # Define the loss function and optimizer
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

    # Train the model
    epochs = 10
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
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
                        help='Dataset to train NN on, one in ["MNIST", "CIFAR10", "CIFAR100"]')
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