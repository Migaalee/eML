# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:53:51 2024

@author: mmiskinyte
"""

### AUX functions for plots


import sys
import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv


"""Load Metrics"""


## Define paths to save files

Metrics="C:/Users/mmiskinyte/Documents/Python_ML/Classifier_metrics/"


LRC05 = "LRC_0_5.csv"

LRC_0_5_metrics = os.path.join(Metrics, LRC05)

LRC_0_5_metrics = pd.read_csv(LRC_0_5_metrics, delimiter=';', index_col=0)



SVC05 = "SVC_0_5.csv"

SVC_0_5_metrics = os.path.join(Metrics, SVC05)

SVC_0_5_metrics = pd.read_csv(SVC_0_5_metrics, delimiter=';', index_col=0)






# Plot
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(df, metric_column, hline_values=None, hline_names=None, hline_styles=None):
    """
    Plot a metric with horizontal lines.

    Parameters:
    - df: DataFrame containing the data.
    - metric_column: The name of the metric column to plot.
    - hline_values: List of values at which to draw horizontal lines.
    - hline_names: List of names for the horizontal lines.
    - hline_styles: List of line styles for the horizontal lines.
    """
    # Check if the metric column exists
    if metric_column not in df.columns:
        raise ValueError(f"Column {metric_column} not found in DataFrame")

    # Define marker and color
    markers = {'yes': '^', 'no': 'o'}  # Triangles for Yes, Circles for No
    colors = ['skyblue', 'seagreen', 'orange']  # Colors for different imputation methods

    # Get unique imputation methods
    imputation_methods = df['Imputation'].unique()

    # Initialize a legend dictionary
    legend_elements = []

    # Adjust figure size
    plt.figure(figsize=(12, 6))  # Width, Height in inches

    # Create a plot for each imputation method
    for i, imputation in enumerate(imputation_methods):
        df_filtered = df[df['Imputation'] == imputation]

        # Plot each point
        for _, row in df_filtered.iterrows():
            plt.scatter(row['Solver_Regularisation'], row[metric_column], 
                        marker=markers[row['Balanced']], 
                        color=colors[i])

        # Add legend entry for imputation method
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', label=imputation,
                          markerfacecolor=colors[i], markersize=10))

    # Add horizontal lines if specified
    hline_legend_elements = []
    if hline_values and hline_names and hline_styles:
        if len(hline_values) == len(hline_names) == len(hline_styles):
            for value, name, style in zip(hline_values, hline_names, hline_styles):
                line = plt.axhline(y=value, linestyle=style, color='black')
                hline_legend_elements.append(plt.Line2D([0], [0], linestyle=style, color='black', label=name))
        else:
            raise ValueError("hline_values, hline_names, and hline_styles must have the same length")

    # Add legend entries for Balanced Yes/No
    for balanced, marker in markers.items():
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', label=f'Balanced = {balanced}',
                                          markerfacecolor='black', markersize=10))

    # Adding labels and title
    plt.xlabel('Solver')
    plt.ylabel(metric_column)
    plt.title(f'Solver Performance by {metric_column}')

    # Rotate x-axis labels
    plt.xticks(rotation=60)

    # Adding custom legend outside the plot
    first_legend = plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1.1), loc='upper left',fontsize='small')
    plt.gca().add_artist(first_legend)
    if hline_legend_elements:
        plt.legend(handles=hline_legend_elements, title='Variant callers', bbox_to_anchor=(1.01, 0.5), loc='center left')

    # Adjust layout to make room for legend
    plt.tight_layout()

    # Show plot
    plt.show()





plot = plot_metric(LRC_0_5_metrics, 'F1',hline_values=[0, 0.012, 0.126], 
                   hline_names=['Bcftools', 'Freebayes', 'Varscan'],  
                   hline_styles=['--', '-.', ':'])


plot




plot_p = plot_metric(LRC_0_5_metrics, 'Precision',hline_values=[0, 0.006, 0.254], 
                   hline_names=['Bcftools', 'Freebayes', 'Varscan'],  
                   hline_styles=['--', '-.', ':'])


plot_p






plot_r = plot_metric(LRC_0_5_metrics, 'Recall',hline_values=[0, 0.586, 0.084], 
                   hline_names=['Bcftools', 'Freebayes', 'Varscan'],  
                   hline_styles=['--', '-.', ':'])


plot_r





plot_err = plot_metric(LRC_0_5_metrics, 'True Error')


plot_err







plotS = plot_metric(SVC_0_5_metrics, 'F1',hline_values=[0, 0.012, 0.126], 
                   hline_names=['Bcftools', 'Freebayes', 'Varscan'],  
                   hline_styles=['--', '-.', ':'])


plotS




plot_pS = plot_metric(SVC_0_5_metrics, 'Precision',hline_values=[0, 0.006, 0.254], 
                   hline_names=['Bcftools', 'Freebayes', 'Varscan'],  
                   hline_styles=['--', '-.', ':'])


plot_pS






plot_rS = plot_metric(SVC_0_5_metrics, 'Recall',hline_values=[0, 0.586, 0.084], 
                   hline_names=['Bcftools', 'Freebayes', 'Varscan'],  
                   hline_styles=['--', '-.', ':'])


plot_rS





plot_errS = plot_metric(SVC_0_5_metrics, 'True Error')


plot_errS










