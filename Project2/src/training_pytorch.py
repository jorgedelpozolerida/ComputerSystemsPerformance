#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for training pytorch Resnet


"""

import os
import sys
import argparse


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

THISFILE_PATH = os.path.abspath(__file__)
DATAIN_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'datain')



# TODO:
# - check that dataset to be used exists
# - train NN


def main(args):

    # different logic depending on input parametrs

    return


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