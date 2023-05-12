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

import warnings
warnings.filterwarnings("ignore")

sns.set_context('paper')


import utils

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


THISFILE_PATH = os.path.abspath(__file__)
PROJECT2_PATH =  os.path.abspath(os.path.join(THISFILE_PATH, os.pardir, os.pardir))

plt.rcParams['lines.linewidth'] = 0.8

'''
TODO:
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
    sns.set(font_scale=1.5)
    sns.lineplot(data=energydata_filtered, x=x_var, y=var_to_plot, ax=ax)
    
    # title and axis management
    ylabels = {
        "power": "Power (W)",
        "temp": "Temperature (ºC)",
        "mem_used": "Memory used (Mib)",
        "gpu_util": "GPU utilization (%)",
        "mem_util": "Memory utilization (%)"
    }
    ax.set_ylabel(ylabels[var_to_plot],  fontsize=18)
    
    
    if x_var == 'time_sec':
        ax.set_xlabel("Time (s)",  fontsize=18)
        ax.set_title(title)
        
        for i, timestamp in enumerate(modeldata_filtered['time_sec']):
            ax.axvline(x=timestamp, color='gray', linestyle='--')

    elif x_var == "epoch_number":
        ax.set_xlabel("Epoch number",  fontsize=18)
        ax.set_title(title)
    
    if var_to_plot in ['gpu_util', 'mem_util']:
        ax.set_ylim(0,100)
    
    ax.tick_params(axis='both', which='major', labelsize=12)

    
    return ax

def plot_singleexperiment_timeseries( levels_values, model_data, energy_data, framework, 
                           vars_toplot = ["power",  "temp",  "mem_used",  "gpu_util",  "mem_util"], 
                           x_var = "time_sec",
                           filename=None,
                           save=True):
    '''
    Function that generates plot for sinlge experiment
    '''

    fig, ax = plt.subplots(1, len(vars_toplot), figsize=(len(vars_toplot)*10, 5))
    
    for i, var_toplot in enumerate(vars_toplot):
        ax[i] = plot_energyvar_timeseries( fig, ax[i],  energy_data, model_data,
                                          levels_values = levels_values,
                                          x_var=x_var,
                                          var_to_plot=var_toplot)    
    if save:
        if filename is None:
            filename = f"singleexperiment_{x_var}_" + "_".join([f"{k}={v}" for k, v in levels_values.items()])+ ".png"
        # save_dir = utils.ensure_dir(os.path.join(PROJECT2_PATH, "plots", framework))
        save_dir = utils.ensure_dir(os.path.join(PROJECT2_PATH, "plots"))
        fig.savefig(os.path.join(save_dir, filename),  bbox_inches='tight')
        
    return fig, ax

def plot_singleexperiment_both_timeseries( levels_values, model_data, energy_data, 
                           vars_toplot = ["power",  "temp",  "mem_used",  "gpu_util",  "mem_util"], 
                           x_var = "time_sec",
                           filename=None,
                           save=True):
    '''
    Function that generates plot for sinlge experiment, both platforms
    '''

    fig, ax = plt.subplots(2, len(vars_toplot) + 1, figsize=(len(vars_toplot)*10, 10),
                           sharex=True,
                           gridspec_kw={'width_ratios': [0.2, 3, 3, 3]})
    

    
    for j, framework in enumerate(model_data.keys()):
 
        for i in range(0, len(vars_toplot) + 1):

            if  i != 0:
                levels_values['framework'] = framework
                ax[j,i] = plot_energyvar_timeseries( fig, ax[j,i],  energy_data[framework], model_data[framework],
                                                levels_values = levels_values,
                                                x_var=x_var,
                                                var_to_plot=vars_toplot[i-1]) 
            else:
                # Add row titles
                ax[j, 0].set_axis_off()
                ax[j, 0].set_title(framework, fontsize=28, x=1, y=0.5, ha='right', va='center', rotation=90)  

            # ax[j,i].set_title(framework, fontsize=20)
    if save:
        if filename is None:
            filename = f"both_{x_var}_" + "_".join([f"{k}={v}" for k, v in levels_values.items()])+ ".png"
        save_dir = utils.ensure_dir(os.path.join(PROJECT2_PATH, "plots"))
        fig.savefig(os.path.join(save_dir, filename),  bbox_inches='tight')
        
    return fig, ax

def plot_3x3(data, filename,  suptitle="",  y_var = 'energy', x_var = 'batch_size', split_var ='framework', columns_var = 'model', rows_var = 'dataset',
             type_plot = 'barplot', hatch = None, ylim=None
                ): 
    '''
    Plots a 3x3 suplots, with barplots split by "var_split", and "x_var" and "y_var".

    Input data from either get_total_energy_data or get_avg_data utputs
    '''
    sns.set(font_scale=1.5)
    
    
    data['energy'] = data['energy']/1000
    
    labels_dict = {
        "power": "Average power (W)",
        "temp": "Temperature (ºC)",
        "mem_used": "Memory used (Mib)",
        "gpu_util": "GPU utilization (in %)",
        "mem_util": "Memory utilization (in %)",
        "energy": "Total energy (kJ)",
        "batch_size": "Batch size",
        "epoch_number": "Epoch",
        "accuracy": "Final accuracy"
    }
    
    
    # get possible column and row values and order them
    if columns_var == 'model':
        column_values = sorted(data[columns_var].unique(), key=lambda x: int(x.split('net')[-1]))
    else:
        column_values = data[columns_var].unique()
        
    row_values = sorted(data[rows_var].unique())
    
    
    fig, axes = plt.subplots(len(row_values), len(column_values) + 1, figsize=(10*len(row_values), 5*len(column_values)), sharey=True, sharex=True, gridspec_kw={'width_ratios': [0.2, 3, 3, 3]})

    for i, row_val in enumerate(row_values):
        
        data_row = data[data[rows_var] == row_val]
        
        
        # Add row titles
        axes[i, 0].set_axis_off()
        axes[i, 0].set_title(row_val, fontsize=28, x=1, y=0.5, ha='right', va='center', rotation=90)
        
        for j, col_value in enumerate(column_values):
            
            col_i = j + 1
            data_column  = data_row[data_row[columns_var] == col_value]
            
            
            if x_var == 'batch_size':
                data_column[x_var] = data_column[x_var].astype(int)
                data_column = data_column.sort_values(by=x_var, ascending=True)

            if type_plot == 'barplot':
                axes[i,col_i] = sns.barplot(x = x_var, y=y_var, hue = split_var, data = data_column, ax=axes[i,col_i], hatch=hatch)
            elif type_plot == 'line_plot':
                axes[i, col_i] = sns.lineplot(x=x_var, y=y_var, hue=split_var, data=data_column, ax=axes[i, col_i])

                
            
            # manage axis labels
            axes[i, col_i].set_axis_on()
            if i == (len(row_values) -1):
                axes[i,col_i].set_xlabel(labels_dict[x_var], fontsize=20)
                if x_var == 'epoch_number':
                    axes[i, col_i].set_xticks(range(1, 11))
            else:
                axes[i, col_i].set_xlabel("")
            if col_i == 1:
                axes[i, col_i].set_ylabel(labels_dict[y_var], fontsize=20)
                axes[i, col_i].tick_params(axis='y', which='both', labelleft=True)
                if ylim is not None:
                    axes[i, col_i].set_ylim(ylim[0], ylim[1])
            else:
                axes[i, col_i].set_ylabel("")
                axes[i, col_i].tick_params(axis='y', which='both', labelleft=False)
            if i == 0:
                axes[i,col_i].set_title(col_value,  fontsize=28)

    fig.suptitle(suptitle, fontsize=24)
    fig.savefig(os.path.join(PROJECT2_PATH, 'plots', filename), bbox_inches='tight')


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
    
    all_data = pd.concat([combined_data_pytorch, combined_data_tensorflow], ignore_index=True)
    all_data.to_csv(os.path.join(PROJECT2_PATH, "dataout", "all_data_processed.csv"), index=False)  
    
    all_data = pd.read_csv(os.path.join(PROJECT2_PATH, "dataout", "all_data_processed.csv" ))

    # PROCESS DATA -------------------------------------------------------------
    
    
    # remove initial epoch due to different behaviour (according to Pina)
    n_epochs_exclude  = 1 # initial epochs to exclude
    all_data_normalepochs = all_data[all_data['epoch_number'] > n_epochs_exclude]
    all_data_firstepochs = all_data[all_data['epoch_number'] <= n_epochs_exclude]

    # total energy
    data_totalenergy_all = utils.get_total_energy_data(all_data)
    
    # average value throughout epochs
    data_avg_all = utils.get_avg_data(all_data)
    data_avg_normalepochs = utils.get_avg_data(all_data_normalepochs)
    data_avg_firstepochs = utils.get_avg_data(all_data_firstepochs)


    # PLOT DATA ------------------------------------------------------------
    
    
    # 1 - Plot a single experiment/combination of levels 
    # NOTE: change values here to select what you want to see
    _logger.info("Generating individual experiment plot")
    levels_values = {
                'device': 'gpu',
                'run': 1,
                'max_epoch': 10,
                'batch_size': 128,
                'framework':  "pytorch",
                'dataset': 'SVHN',
                'model': 'resnet101'
                }
    # pytorch
    plot_singleexperiment_timeseries(levels_values,  model_data_pytorch, energy_data_pytorch, framework=levels_values['framework'],
                                        vars_toplot = ["power",  "temp", "gpu_util"],
                                        x_var="time_sec",
                                        filename = f"singleexperiment_levels_{levels_values['run']}-{levels_values['framework']}-{levels_values['batch_size']}-{levels_values['dataset']}-{levels_values['model']}.png" 
                                        )
    # tensorflow
    levels_values['framework'] = 'tensorflow'
    plot_singleexperiment_timeseries(levels_values,  model_data_tensorflow, energy_data_tensorflow, framework=levels_values['framework'],
                                        vars_toplot = ["power",  "temp", "gpu_util"],
                                        x_var="time_sec",
                                         filename = f"singleexperiment_levels_{levels_values['run']}-{levels_values['framework']}-{levels_values['batch_size']}-{levels_values['dataset']}-{levels_values['model']}.png" 
                                        )
    # combined
    plot_singleexperiment_both_timeseries(levels_values,  {'pytorch':model_data_pytorch, 'tensorflow':model_data_tensorflow}, {'pytorch':energy_data_pytorch, 'tensorflow':energy_data_tensorflow},
                                    vars_toplot = ["power",  "temp", "gpu_util"],
                                    x_var="time_sec",
                                    filename =f"singleexperiment_bothframeworks_levels_{levels_values['run']}-{levels_values['batch_size']}-{levels_values['dataset']}-{levels_values['model']}.png")
    
    # 2 - Total energy consumption of training between the frameworks per Model and Batch size
    # NOTE: after inspecting single experiments, I see that first epoch can be removed, but maybe i'm wrong
    
    
    # Generate plot for total energy
    _logger.info("Generating total energy plot")
    plot_3x3(data = data_totalenergy_all, filename=f'total_energy_per_batchsize.png',
             y_var = 'energy', x_var = 'batch_size', split_var ='framework', columns_var = 'model', rows_var = 'dataset',
             type_plot = 'barplot'
             )

    # 3 - Average power consumption 
    _logger.info("Generating power through epochs plots")
    dataset = 'SVHN'
    data_plot = all_data[all_data['dataset'] == dataset] # chose only one
    plot_3x3(data = data_plot, filename=f'power_through_epochs_{dataset}.png',
             y_var = 'power', x_var = 'epoch_number', split_var ='framework', columns_var = 'model', rows_var = 'batch_size',
             type_plot = 'line_plot'
             )
    # 4- Mean energy graph for 1st epoch and other epochs
    _logger.info("Generating average power through epochs plots")
    # all epochs
    plot_3x3(data = data_avg_all, filename=f'averagepower_per_batchsize.png',
             y_var = 'power', x_var = 'batch_size', split_var ='framework', columns_var = 'model', rows_var = 'dataset',
             type_plot = 'barplot', hatch='/'
             )

    # 5 - Make mean GPU utilization graph
    _logger.info("Generating average GPU utilization through epochs plots")
    # all epochs
    plot_3x3(data = data_avg_all, filename=f'averageGPUutilization_per_batchsize.png',
             y_var = 'gpu_util', x_var = 'batch_size', split_var ='framework', columns_var = 'model', rows_var = 'dataset',
             type_plot = 'barplot', hatch='x', ylim=[0, 100]
             )
    # 6 - Make accuraccy graph
    _logger.info("Generating final accuracy plot")
    # last epoch
    all_data_lastepoch = all_data[all_data['epoch_number'] == 10]
    plot_3x3(data = all_data_lastepoch, filename=f'final_accuracy_per_batchsize.png',
             y_var = 'accuracy', x_var = 'batch_size', split_var ='framework', columns_var = 'model', rows_var = 'dataset',
             type_plot = 'barplot', hatch='.', ylim=[0, 1]
             )

    # 4 - Maximum energy spike of the same variance
    # 5 - Growth of energy usage + gpu utilization for increase in batch size
    # 6 - Mean gpu utilization between frameworks


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