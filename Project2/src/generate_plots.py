#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for generating experiment plots

Select either cpu or gpu

NOTE: 
- this file is assumed to be placed under Project2/src inside repo
- make sure you only contain files with newest filename structure
- unique id for some time series of data is combinaiton of: run, device, max_epoch, batch_size, framework, dataset, model
"""

import os
import sys
import argparse


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns



import utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


THISFILE_PATH = os.path.abspath(__file__)
PROJECT2_PATH =  os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir))


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
    if framework == 'pytorch':
        # read with error in it
        header  = ["epoch","precision","recall","f1", "error", "timestamp"]
        model_data = pd.read_csv(csv_path, skiprows=[0], sep=';|;;', names=header, engine='python')
        model_data.drop("error", axis=1, inplace=True)
    else :
        # read normally
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

def get_allexperiments_data(exp_dir, framework):
    '''
    Generates necessary data for framework experiments
    '''
    
    # Generate overview of data for framwork and save
    df_overview = get_experiments_overview(exp_dir, framework)
    save_dir = utils.ensure_dir(os.path.join(PROJECT2_PATH,"dataout", framework ))

    
    
    # Extract tabular data for energy and model separately

    energy_data = []
    model_data = []
    
    for index, row in tqdm(df_overview.iterrows(), desc=f"Reading csv data from {framework} experiments", total=df_overview.shape[0] ):
        # read data for that experiment
        energy_data_temp = read_energy_csv(row['energy_filepath'])
        model_data_temp = read_model_csv(row['model_filepath'], framework,
                                         timestamp_0 = energy_data_temp['timestamp'].min() # needed to coordinate both time series
                                         )
        row_temp = row.drop(['energy_filepath', 'model_filepath'])
        
        # Add id columns and append to list with all data
        model_data.append(model_data_temp.assign(**row_temp))
        energy_data.append(energy_data_temp.assign(**row_temp))


    model_data = pd.concat(model_data)
    energy_data = pd.concat(energy_data)
    
    
    _logger.info(f"Saving overview for {framework}")
    df_overview.drop(['energy_filepath', 'model_filepath'], axis=1).to_csv(os.path.join(save_dir, f"data_overview_{framework}.csv"), 
                       index=False)

    return df_overview, model_data, energy_data


def plot_energyvar_timeseries(fig, ax, energy_data, model_data, levels_values, var_to_plot, title=""):
    '''
    Plots some var_to_plot 'y' through time from energy data for some combination of levels values (id).
    
    note: example for 'levels_values': 
    {
    'run': 1,
    'device': 'gpu',
    'max_epoch': 10,
    'batch_size': 64,
    'framework':  'pytorch',
    'dataset': 'CIFAR100',
    'model': 'resnet101'
    }
    '''
    # timestamp  power  temp  mem_used  gpu_util  mem_util  time_sec run device max_epoch batch_size framework   dataset      model

    assert len(levels_values.keys()) == 7, "Please provide a value for all levels"
    
    # filter data
    query_string = ' & '.join(f'{k} == "{v}"' for k, v in levels_values.items())
    energydata_filtered = energy_data.query(query_string)
    modeldata_filtered = model_data.query(query_string)

    # plot data
    sns.lineplot(data=energydata_filtered, x="time_sec", y=var_to_plot, ax=ax)
    
    # title and axis management
    ylabels = {
        "power": "Power consumption (W)",
        "temp": "Temperature (ºC)",
        "mem_used": "Memory used (Mib)",
        "gpu_util": "GPU utilization (in %)",
        "mem_util": "Memory utilization (in %)"
    }
    ax.set_ylabel(ylabels[var_to_plot])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    
    for i, timestamp in enumerate(modeldata_filtered['time_sec']):
        ax.axvline(x=timestamp, color='gray', linestyle='--')
        # ax.text(timestamp, 0.95, f"batch:{i+1}", 
        #         rotation=90, 
        #         ha='right', va='top', 
        #         fontsize=8)
    
    
    return ax

def plot_single_experiment( levels_values, model_data, energy_data, framework, 
                           vars_toplot = ["power",  "temp",  "mem_used",  "gpu_util",  "mem_util"], 
                           save=True):
    '''
    Function that generates plot for sinlge experiment
    '''

    fig, ax = plt.subplots(1, len(vars_toplot), figsize=(len(vars_toplot)*10, 5))
    title =  "Experiment with parameters: " + ", ".join([f"{k}={v}" for k, v in levels_values.items()])
    
    for i, var_toplot in enumerate(vars_toplot):
        ax[i] = plot_energyvar_timeseries( fig, ax[i],  energy_data, model_data,
                            levels_values = levels_values,
                        var_to_plot=var_toplot)
    
    fig.suptitle(title)
    
    if save:
        save_dir = utils.ensure_dir(os.path.join(PROJECT2_PATH, "plots", framework))
        fig.savefig(os.path.join(save_dir, "singleexperiment_" + "_".join([f"{k}={v}" for k, v in levels_values.items()])+ ".png"),  bbox_inches='tight')


def main(args):


    exp_dir = os.path.join(PROJECT2_PATH, "experiments", args.device)
    
    # PYTORCH -----------------------------------------------------------------
    # Load experiment data
    df_overview_pytorch, model_data_pytorch, energy_data_pytorch = get_allexperiments_data(exp_dir, "pytorch")
    print("PYTORCH\n",
          f"Number of experiments: {df_overview_pytorch.shape}\n", 
          f"Size of model data: {model_data_pytorch.shape}\n", 
          f"Size of energy data: {energy_data_pytorch.shape}"
          )


    # 1 - Single experiment ---------------------------------------------
    # NOTE: change values here to select what you want to see
    levels_values = {
                'run': 1,
                'device': 'gpu',
                'max_epoch': 10,
                'batch_size': 128,
                'framework':  "pytorch",
                'dataset': 'SVHN',
                'model': 'resnet50'
                }
    plot_single_experiment(levels_values, model_data_pytorch, energy_data_pytorch, framework='pytorch',
                           vars_toplot = ["power",  "temp", "gpu_util",  "mem_util"])
    
    


    # TENSORFLOW -----------------------------------------------------------------


    




    return


def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='gpu',
                        help='Device: cpu or gpu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)