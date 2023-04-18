#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script that downloads preprocessed data ready to be used for training


It downloads all datasets and:
- does train-test splits
- preprocesses each dataset for each ResNet

Only needs to be run once and takes functions from utils script.
"""


import os
import sys
import argparse
from utils import generate_trainingdata


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

current_file_path = os.path.abspath(__file__)
data_path = os.path.join( os.path.abspath(os.path.join(current_file_path, os.pardir, os.pardir)), 'data')

def main(args):

    if args.out_dir is None:
        out_path = data_path
    else:
        out_path = os.path(args.out_dir)
        
    generate_trainingdata(out_path)

    return


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_dir', type=str, default=None, required=False,
                        help='Path to the output directory')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)