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


'''
TODO:
- Total energy consumption of training between the frameworks per Model and Batch size
- Maximum energy spike of the same variance
- Growth of energy usage + gpu utilization for increase in batch size
- Mean gpu utilization between frameworks
'''


def plot_energyvar_timeseries(fig, ax, energy_data, model_data, levels_values, var_to_plot,x_var="time_sec", title=""):
    '''
    Plots some var_to_plot 'y' through time from energy data for some combination of levels values (id).
    
    note: example for 'levels_values': 
    {
    'run': 1, # not necessary if per epochs
    'device': 'gpu',
    'max_epoch': 10,
    'batch_size': 64,
    'framework':  'pytorch',
    'dataset': 'CIFAR100',
    'model': 'resnet101'
    }
    '''    
    # filter data
    query_string = ' & '.join(f'{k} == "{v}"' for k, v in levels_values.items())
    energydata_filtered = energy_data.query(query_string)
    modeldata_filtered = model_data.query(query_string)

    # plot data
    sns.lineplot(data=energydata_filtered, x=x_var, y=var_to_plot, ax=ax)
    
    # title and axis management
    ylabels = {
        "power": "Power consumption (W)",
        "temp": "Temperature (ºC)",
        "mem_used": "Memory used (Mib)",
        "gpu_util": "GPU utilization (in %)",
        "mem_util": "Memory utilization (in %)"
    }
    ax.set_ylabel(ylabels[var_to_plot])
    
    
    if x_var == 'time_sec':
        ax.set_xlabel("Time (s)")
        ax.set_title(title)
        
        for i, timestamp in enumerate(modeldata_filtered['time_sec']):
            ax.axvline(x=timestamp, color='gray', linestyle='--')

    elif x_var == "epoch_number":
        ax.set_xlabel("Epoch number")
        ax.set_title(title)
        
    
    return ax

def plot_singleexperiment_timeseries( levels_values, model_data, energy_data, framework, 
                           vars_toplot = ["power",  "temp",  "mem_used",  "gpu_util",  "mem_util"], 
                           x_var = "time_sec",
                           save=True):
    '''
    Function that generates plot for sinlge experiment
    '''

    fig, ax = plt.subplots(1, len(vars_toplot), figsize=(len(vars_toplot)*10, 5))
    title =  f"Experiment through {x_var} with parameters: " + ", ".join([f"{k}={v}" for k, v in levels_values.items()])
    
    for i, var_toplot in enumerate(vars_toplot):
        ax[i] = plot_energyvar_timeseries( fig, ax[i],  energy_data, model_data,
                                          levels_values = levels_values,
                                          x_var=x_var,
                                          var_to_plot=var_toplot)
    
    fig.suptitle(title)
    
    if save:
        save_dir = utils.ensure_dir(os.path.join(PROJECT2_PATH, "plots", framework))
        fig.savefig(os.path.join(save_dir, f"singleexperiment_{x_var}_" + "_".join([f"{k}={v}" for k, v in levels_values.items()])+ ".png"),  bbox_inches='tight')

def plot_singleexperiment_epochs( levels_values, combined_data, framework, 
                           vars_toplot = ["power",  "temp",  "mem_used",  "gpu_util",  "mem_util"], 
                           x_var = "epoch_number",
                           save=True):
    '''
    Function that generates plot for sinlge experiment
    '''

    fig, ax = plt.subplots(1, len(vars_toplot), figsize=(len(vars_toplot)*10, 5))
    title =  f"Experiment through {x_var} with parameters: " + ", ".join([f"{k}={v}" for k, v in levels_values.items()])
    
    for i, var_toplot in enumerate(vars_toplot):
        ax[i] = plot_energyvar_timeseries( fig, ax[i],  energy_data = combined_data, model_data = None,
                                          levels_values = levels_values,
                                          x_var=x_var,
                                          var_to_plot=var_toplot)
    
    fig.suptitle(title)
    
    if save:
        save_dir = utils.ensure_dir(os.path.join(PROJECT2_PATH, "plots", framework))
        fig.savefig(os.path.join(save_dir, f"singleexperiment_{x_var}_" + "_".join([f"{k}={v}" for k, v in levels_values.items()])+ ".png"),  bbox_inches='tight')

def main(args):
    
    # LOAD DATA ------------------------------------------------------------
    # Pytorch 
    
    df_overview_pytorch, model_data_pytorch, energy_data_pytorch, combined_data_pytorch = utils.get_allexperiments_data("pytorch", device=args.device)
    print("PYTORCH\n",
          f"Number of experiments: {df_overview_pytorch.shape}\n", 
          f"Size of model data: {model_data_pytorch.shape}\n", 
          f"Size of energy data: {energy_data_pytorch.shape}\n",
          f"Size of combined data per epoch (averaged through runs): {combined_data_pytorch.shape}",
          )
    # Tensorflow ------------------------------------------------------------
    df_overview_tensorflow, model_data_tensorflow, energy_data_tensorflow, combined_data_tensorflow = utils.get_allexperiments_data( "tensorflow", device=args.device)
    print("TENSORFLOW\n",
          f"Number of experiments: {df_overview_tensorflow.shape}\n", 
          f"Size of model data: {model_data_tensorflow.shape}\n", 
          f"Size of energy data: {energy_data_tensorflow.shape}\n",
          f"Size of combined data per epoch (averaged through runs): {combined_data_tensorflow.shape}",

          )
    
    # All data merged together and saved 
    combined_data_pytorch['framework'] = 'pytorch'
    combined_data_tensorflow['framework'] = 'tensorflow'
    
    merged_df = pd.concat([combined_data_pytorch, combined_data_tensorflow], ignore_index=True)
    merged_df.to_csv(os.path.join(PROJECT2_PATH, "dataout", "all_data_processed.csv"), index=False)


    # # 1 - Single experiment
    # # NOTE: change values here to select what you want to see
    # levels_values = {
    #             'run': 1,
    #             'device': 'gpu',
    #             'max_epoch': 10,
    #             'batch_size': 128,
    #             'framework':  "pytorch",
    #             'dataset': 'SVHN',
    #             'model': 'resnet50'
    #             }
    # # plot single run for a combinaiton of levels
    # plot_singleexperiment_timeseries(levels_values, model_data_pytorch, energy_data_pytorch, framework='pytorch',
    #                                     x_var="time_sec",
    #                                     vars_toplot = ["power",  "temp", "gpu_util",  "mem_util"])
    # levels_values.pop('run')
    # # plot avergaed runs averaged per epoch for combinaiton of levels
    # plot_singleexperiment_timeseries(levels_values, model_data_pytorch, combineddata_perepoch_averaged, framework='pytorch',
    #                                     vars_toplot = ["power",  "temp", "gpu_util",  "mem_util"],
    #                                     x_var="epoch_number"
    #                                     )
    
    # 2 - Total energy consumption of training between the frameworks per Model and Batch size
    # 3 - Maximum energy spike of the same variance
    # 4 - Growth of energy usage + gpu utilization for increase in batch size
    # 5 - Mean gpu utilization between frameworks


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