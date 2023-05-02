#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training tensorflow Resnets


"""

import os
import argparse
from utils import get_dataset, get_modeloutputdata, write_to_file

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'datain')
EXPORT_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tuple_map = lambda x, y: (x, y)


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_x, test_y, csv_path):
        super().__init__()
        self.test_x = test_x
        self.test_y = test_y
        self.csv_path = csv_path

    def on_epoch_end(self, epoch, logs=None):
        score = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        print(score)

def main(args):
    file_name  = f"run{args.run}-{args.device}-epoch{args.epochs}-batchsize{args.batch_size}-tensorflow-{args.dataset}-{args.resnet_size}_MODEL.csv"
    csv_path = os.path.join(EXPORT_PATH, 'experiments', args.device, 'tensorflow', file_name)

    # Print the parsed arguments
    print(f"Framework: Tensorflow")
    print(f"Dataset: {args.dataset}")
    print(f"Resnet size: {args.resnet_size}")
    # print(f"Epochs: {args.epochs}")
    # print(f"Batch size: {args.batch_size}")
    # print(f"Output directory: {args.out_dir}")
    print("**************************************")

    # Get all prints before training away
    # output_buffer = io.StringIO()
    
    # Handle device used
    print(tf.config.list_physical_devices())
    assert args.device in ['cpu', 'gpu'], "Please select only 'cpu' or 'gpu' as device"
    if args.device == 'gpu':
        physical_devices = tf.config.list_physical_devices('GPU') # get number of gpu
        # check if GPU is available
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("GPU: ",physical_devices)
        else:
            print(tf.config.list_physical_devices())
            #raise ValueError("There is no gpu available, please use cpu")

    # Load the dataset
    (x_train, y_train), (x_test, y_test) = get_dataset(args.dataset)

    num_classes = len(np.unique(y_train))
    
    # Define ResNet
    shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])    
    if args.resnet_size == "resnet50":
        resnet = tf.keras.applications.ResNet50(
            include_top=False, weights=None, input_shape=shape, pooling="avg", classes=num_classes
        )
    elif args.resnet_size == "resnet101":
        resnet = tf.keras.applications.ResNet101(
            include_top=False, weights=None, input_shape=shape, pooling="avg", classes=num_classes
        )
    elif args.resnet_size == "resnet152":
        resnet = tf.keras.applications.ResNet152(
            include_top=False, weights=None, input_shape=shape, pooling="avg", classes=num_classes
        )
    else:
        raise ValueError("Invalid ResNet model name")

    # Prepare the training dataset
    batch_size = int(args.batch_size)

    inputs = tf.keras.Input(shape=shape)
    x = resnet(inputs, training=True)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    # Define the loss function and optimizer
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[tf.keras_metrics.precision(), 
                 tf.keras_metrics.recall(),
                 tf.keras.metrics.Accuracy(),
                 tf.keras.metrics.F1Score(average='macro')
        ]
    )

    #write_to_file("epoch;precision;recall;accuracy;f1;timestamp", csv_path)
    
    # start training
    model.fit(np.asarray(x_train), np.asarray(y_train), 
              epochs=int(args.epochs), 
              batch_size=batch_size, 
              steps_per_epoch=100, 
              callbacks=[CustomCallback(x_test, y_test, csv_path)])


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
