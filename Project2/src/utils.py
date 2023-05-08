#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Script with functions to be used across files
"""

import os

from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import numpy as np                                                                  # NOQA E402
import datetime
import pandas as pd
import logging
from tqdm import tqdm
import re

# constant which denotes where to cache the data
THISFILE_PATH = os.path.abspath(__file__)
PROJECT2_PATH =  os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir))
DATAIN_PATH = os.path.join( os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir)), 'datain')

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


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


def get_modeloutputdata(values):
    '''
    Returns a formated row to be added to csv
    '''
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    value_str = ""

    for val in values:
        value_str = value_str + f"{val};"

    return f"{value_str}{timestamp}"

def write_to_file(data: str, path: str):
    with open(path, "a+") as file:
        file.write(data+"\n")

def generate_metrics(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    percision = precision_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average='macro')

    return (acc, recall, percision, f1)

def ensure_dir(directory):
    '''
    Creates dir if it does not exist and returns it
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created the following dir: {directory}")
    
    return directory


###########################################
# ------ Experiment data loading ---------
###########################################



def get_modelpath_from_energypath(path):
    '''
    Creates full path for model path based on energy path
    '''
    newpath = path.replace("-ENERGY", "_MODEL")
    if os.path.exists(newpath):
        return newpath
    else:
        return  None

def get_experiments_overview(exp_dir, framework):
    '''
    Gets an overview of available data with necessary paths
    '''
    csv_list = [os.path.join(exp_dir, framework, f) for f in os.listdir(os.path.join(exp_dir, framework)) if f.endswith('.csv')]

    df_list = []
    
    # Parse 
    for csv_fullpath in csv_list:
        if csv_fullpath.endswith("ENERGY.csv"):
            filename = os.path.basename(csv_fullpath)
            file_parts = re.split('-', filename)
            sub_df = {
            "run": re.findall(r'\d+', file_parts[0])[0],
            "device": file_parts[1],
            "max_epoch": re.findall(r'\d+', file_parts[2])[0],
            "batch_size":  re.findall(r'\d+', file_parts[3])[0],
            "framework": file_parts[4],
            "dataset": file_parts[5],
            "model": file_parts[6],
            "energy_filepath": csv_fullpath,
            "model_filepath": get_modelpath_from_energypath(csv_fullpath)
            }
            df_list.append(sub_df)
            
    df = pd.DataFrame.from_records(df_list)
                
    return df


def read_energy_csv(csv_path):
    '''
    Reads energy data an processes it into clean format
    '''
    energy_data = pd.read_csv(csv_path)
    
    # rename cols
    new_column_names = {
    "timestamp": "timestamp",
    ' power.draw [W]': 'power',
    ' temperature.gpu': 'temp',
    ' memory.used [MiB]': 'mem_used',
    ' utilization.gpu [%]': 'gpu_util',
    ' utilization.memory [%]': 'mem_util'
    }
    energy_data.rename(columns=new_column_names, inplace=True)

    # Process data into nice format and datatype
    # time
    energy_data['timestamp'] = pd.to_datetime(energy_data['timestamp'])
    energy_data['time_sec'] = (energy_data['timestamp'] - energy_data['timestamp'][0]).dt.total_seconds()
    # power
    energy_data['power'] = energy_data['power'].str.strip('W').astype(float)  
    # memory
    energy_data['mem_used'] = energy_data['mem_used'].str.strip('MiB').astype(float)
    # percentages
    energy_data['gpu_util'] = energy_data['gpu_util'].str.strip('%').astype(int)
    energy_data['mem_util'] = energy_data['mem_util'].str.strip('%').astype(int)

    return energy_data

def read_model_csv(csv_path, framework, timestamp_0):
    '''
    Reads model data an processes it into clean format
    '''
    # if framework == 'pytorch':
    #     # read with error in it
    #     header  = ["epoch","precision","recall","accuracy", "f1", "error", "timestamp"] 
    #     model_data = pd.read_csv(csv_path, skiprows=[0], sep=';|;;', names=header, engine='python')
    #     model_data.drop("error", axis=1, inplace=True)
    # else :
    #     continue
    #     # read normally
    
    model_data = pd.read_csv(csv_path, sep=";")
    
    # Process data into nice format and datatype
    # time
    model_data['timestamp'] = pd.to_datetime(model_data['timestamp'])
    model_data['time_sec'] = (model_data['timestamp'] - timestamp_0).dt.total_seconds()
    # round evaluation metrics
    model_data['precision'] = model_data['precision'].round(3)
    model_data['recall'] = model_data['recall'].round(3)
    model_data['f1'] = model_data['f1'].round(3)
    
    return model_data

def get_allexperiments_data(framework, device="gpu"):
    '''
    Generates necessary data for framework experiments
    '''
    exp_dir = os.path.join(PROJECT2_PATH,  "experiments", device)
    # Generate overview of data for framwork and save
    df_overview = get_experiments_overview(exp_dir, framework)
    save_dir = ensure_dir(os.path.join(PROJECT2_PATH,"dataout", framework ))

    
    
    # Extract tabular data for energy and model separately

    energy_data = []
    model_data = []
    
    for index, row in tqdm(df_overview.iterrows(), desc=f"Reading csv data from {framework} experiments", total=df_overview.shape[0] ):
        # read data for that experiment
        energy_data_temp = read_energy_csv(row['energy_filepath'])
        model_data_temp = read_model_csv(row['model_filepath'], framework,
                                         timestamp_0 = energy_data_temp['timestamp'].min() # needed to coordinate both time series
                                         )

        # Assign bins to time in energy data 
        bins = model_data_temp['time_sec'].unique().tolist()
        bins = [-1] + bins
        bins[-1] = bins[-1]
        labels = [int(i) + 1 for i in model_data_temp['epoch'].unique().tolist()]


        energy_data_temp['epoch_number'] =  pd.cut( energy_data_temp['time_sec'], bins=bins, labels=labels)

        
        # remove data not inside any epoch timeframe
        energy_data_temp = energy_data_temp[ ~ energy_data_temp['epoch_number'].isna()]
        
        # convert all epochs to same scale
        model_data_temp['epoch'] = model_data_temp['epoch'].astype(int) + 1


        # Add id columns and append to list with all data
        row_temp = row.drop(['energy_filepath', 'model_filepath'])

        model_data.append(model_data_temp.assign(**row_temp))
        energy_data.append(energy_data_temp.assign(**row_temp))


    model_data = pd.concat(model_data)
    energy_data = pd.concat(energy_data)
    
    # combine data per runs and per batches and average across runs
    columns_to_avg = [
        # energy cols
        'power', 'temp', 'mem_used', 'gpu_util', 'mem_util',
        # model cols
        "precision", "recall", "accuracy", "f1"
                      ]
    energy_data_combined = pd.merge(energy_data,
                                    model_data,
                                    left_on=['epoch_number', 'run', 'device', 'max_epoch', 'batch_size', 'framework','dataset', 'model'],
                                    right_on = ['epoch', 'run', 'device', 'max_epoch', 'batch_size', 'framework','dataset', 'model'],
                                    how='left')
    energydata_averagedperepoch = energy_data_combined.groupby(['epoch_number', 'run', 'device', 'max_epoch', 'batch_size', 'framework', 'dataset', 'model'])[columns_to_avg].agg('mean').reset_index()
    energydata_averagedperepoch_averagedthroughruns = energydata_averagedperepoch.groupby(['epoch_number', 'device', 'max_epoch', 'batch_size', 'framework', 'dataset', 'model'])[columns_to_avg].agg('mean').round(3).reset_index()


    # Save data
    _logger.info(f"Saving overview for {framework}")
    df_overview.drop(['energy_filepath', 'model_filepath'], axis=1).to_csv(os.path.join(save_dir, f"data_overview_{framework}.csv"), 
                       index=False)
    energydata_averagedperepoch_averagedthroughruns.to_csv(os.path.join(save_dir, f"cleandata_perepoch_{framework}.csv"), index=False)

    return df_overview, model_data, energy_data, energydata_averagedperepoch_averagedthroughruns
