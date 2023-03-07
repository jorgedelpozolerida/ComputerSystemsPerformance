#!/usr/bin/env python
# -*-coding:utf-8 -*-
""" Script for generating plots for all experiments combined


{Long Description of Script}
"""
__author__ = "Jorge del Pozo Lérida"
__email__ =  "jorgedelpozolerida@gmail.com"
__date__ =  "16/02/2023"


import os
import glob
import argparse


import logging                                                                      # NOQA E402
import numpy as np                                                                  # NOQA E402
import pandas as pd                                                                 # NOQA E402
import matplotlib.pyplot as plt



logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# Plotting config
PLOT_MARKERS = ["|", "x", "2", "s", "P", "o"]
PLOT_COLORS = ['r', 'g', 'b', 'm', 'c', 'y']
plt.rcParams.update({'font.size': 16})
plt.rcParams['lines.linewidth'] = 0.8

# TO DO:
# - Include memcpy for comparison (extreme upper bound)
# - Change code if different number of hash bits

def process_single_experiment_run(csv_path, args):
    '''
    Process whatever is on csv form experiment. 
    Expected columns: Threads;Hash_Bits;Running Time (ms)
    '''
    input_size = int(args.input_size)
    df = pd.read_csv(csv_path, sep=';', index_col=False).rename(
        columns={ 
                "Threads": "threads",
                "Hash_Bits": "bits",
                "Running Time (ms)": "miliseconds"
                })
    df['throughput'] = (input_size/ df['miliseconds']) / 1000 # to have millions/second 

    return df


def combine_experiment_trials(experiment_dir, args):
    '''
    Average over experiment runs
    '''
    all_files = glob.glob(os.path.join(experiment_dir, "*.csv"))
    all_dfs = [process_single_experiment_run(os.path.join(experiment_dir, f), args) for f in all_files]
    data_averaged = pd.concat(all_dfs).groupby(['bits', 'threads']).mean().reset_index()
    return data_averaged


def plot_experiment(ax, data_technique, title):
    '''
    Plot data for an experiment on given axis
    '''
    # Extract values for plot
    thread_values = data_technique['threads'].unique()
    labels = [f"Threads: {t}" for t in thread_values]
    max_bits = data_technique['bits'].max()
    ticks = np.arange(0, max_bits + 2, step=2)
    assert len(PLOT_MARKERS) >= len(thread_values), \
        "Need to specify more markers and colors for threads in PLOT_MARKERS PLOT_COLORS "
    # Set axes values
    ax.set_title(f"{title}")
    ax.set_ylabel("Millions of Tuples per Second")
    ax.set_xlabel("Hash Bits")
    ax.set_xticks(ticks=ticks, labels=ticks)

    # Plot
    for j, th in enumerate(thread_values):
        thread_data = data_technique.query("threads == @th")
        ax.plot(thread_data['bits'], thread_data['throughput'], 
            label=labels[j], 
            marker=PLOT_MARKERS[j % len(PLOT_MARKERS)], 
            color=PLOT_COLORS[j % len(PLOT_COLORS)])
        ax.legend(labels,  prop={'size': 12})

    return ax


def main(args):

    input_dir = os.path.abspath(args.in_dir)
    output_dir = os.path.abspath(args.out_dir)

    experiment_dirs = os.listdir(input_dir)
    experiment_dirs.sort(reverse=True)
    
    # Get all data
    data_techniques = [combine_experiment_trials(os.path.join(input_dir, subdir), args) for subdir in experiment_dirs ]
    
    # Big plot config
    fig_all, axs_all = plt.subplots(1, len(experiment_dirs), figsize=(15,5), sharex=True)
    fig_all.subplots_adjust(hspace=0)
    y_lims = [0, np.ceil(max(  [ d['throughput'].max() for d in data_techniques] ))]

    # Plot
    for i, technique_dir  in enumerate(experiment_dirs):

        data_technique = data_techniques[i]

        # Single plot creation and saving
        fig, ax = plt.subplots(figsize=(8,5))
        ax = plot_experiment(ax,  data_technique, title=technique_dir)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{technique_dir}_plot.png" ))

        # Subplot
        axs_all[i] = plot_experiment(axs_all[i], data_technique, title=technique_dir)
        axs_all[i].set_ylim(y_lims)
        
        _logger.info( f"Generated plots for: {technique_dir }")
    
    # Big plot saving
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(output_dir, f"all_experiments_plot.png" ))



def parse_args():
    '''
    Parses all script arguments.
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir', type=str, default=None,
                        help='Path to the input directory, expecting one folder per algorithm ')
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Path to the output directory where plots are saved')
    parser.add_argument('--input_size', type=str, default=None,
                        help='Number of tuples/data size used in experiment')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)