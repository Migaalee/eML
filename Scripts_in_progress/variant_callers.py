# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:46:51 2023

@author: mmiskinyte
"""

import sys
import os
import io
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
import itertools
from bokeh.layouts import gridplot
import aplanat
from aplanat import bars
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import hstack

import re

### AUX functions

def parse_vcf(fname, info_cols=None, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP':int,'CIGAR':str,}, nrows=1000)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    # create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    if info_cols is not None:
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    return vcf


def parse_vcf2(fname, info_cols=None, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP':int,'CIGAR':str,}, nrows=1000)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT GT2".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    # create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";") \
        .apply(lambda x: dict([y.split("=") for y in x]))
    if info_cols is not None:
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    return vcf




def match_groudtruth (dataset, gt_dataset):
  """ Return dataset with ground truth data (1 for variant and 0 for false positive)
  based on the provided gt_dataset. Return dataset only with continous features. This function
  is to be used for creating dataset for training with ML classifiers.
 """
  dataset2=dataset.assign(GROUND=dataset.POS.isin(gt_dataset.POS).astype(int))
  dataset3= dataset2.loc[:,((dataset2.dtypes=='float64').values | (dataset2.dtypes=='int').values)]
  dataset4 = dataset3.set_index('POS')
  return dataset4


def confusion_matrix_m(dataset, gt_dataset, genome_size):
    """
    Based on the dataset that was generated from variant caller pipeline and ground truth dataset
    (all known mutations, e.g., all true positives) calculate the confusion matrix.
    """
    d_P = dataset["POS"].values
    g_P = gt_dataset["POS"].values
    matrix = np.zeros((2, 2))  # Matrix of 2 by 2 (added a missing closing parenthesis)
    matrix[1, 1] = genome_size - len(gt_dataset)  # True Negatives
    for i in d_P:
        if i in g_P:
            matrix[0, 0] += 1  # True Positives
        elif i not in g_P:
            matrix[0, 1] += 1  # False Positives
    for k in g_P:
        if k not in d_P:
            matrix[1, 0] += 1  # False Negatives
    return matrix





def plot_custom_confusion_matrix(confusion_matrix, title="Conf Matrix"):
    """
    This function plots a custom 2x2 confusion matrix with specified colors by name and labels for TP, TN, FP, FN.

    Args:
    confusion_matrix (list of lists): A 2x2 confusion matrix in the format [[TP, FP], [FN, TN]]
    title (str): Custom title for the confusion matrix (optional)

    Returns:
    None
    """
    class_names = ['Positive', 'Negative']

    plt.imshow(confusion_matrix, interpolation='nearest', cmap='gray', vmin=0, vmax=1, aspect='auto')  # Light gray background
    plt.title(title)  # Set the custom title

    tick_marks = [0, 1]
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    labels = ["TP", "FP", "FN", "TN"]
    for i in range(2):
        for j in range(2):
            if i == j:
                color = 'green'  # TP and TN in limegreen
            else:
                color = 'red'    # FP and FN in pink
            plt.text(j, i, str(confusion_matrix[i][j]), horizontalalignment="center", color=color)
            plt.text(j, i-0.2, labels[i*2+j], horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Example usage with a custom title:
# Replace the example matrix with your actual 2x2 confusion matrix
# custom_matrix = [[42, 8], [12, 38]]
# plot_custom_confusion_matrix(custom_matrix, title="Confusion Matrix - My Title")




def metrics_m(matrix, calculate=None):
  """Function to calculate metrics for different variant callers based on confusion matrix.
  If no second argument is given, returns all the metrics. If particular metric is necessary,
  define the argument in the following way.
  Example:
  precision=metrics(matrix, "precision")
  """
  precision=matrix[0,0]/(matrix[0,0]+matrix[0,1])
  recall= matrix[0,0]/(matrix[0,0]+matrix[1,0])
  accuracy=(matrix[0,0]+matrix[1,1])/(matrix[0,0]+matrix[0,1]+matrix[1,0]+matrix[1,1])
  f1=2*(recall*precision)/(recall+precision)
  metrics_m=[precision, recall, accuracy, f1]
  if calculate is None:
    return metrics
  elif calculate == "precision":
    return precision
  elif calculate == "recall":
    return recall
  elif calculate == "accuracy":
    return accuracy
  elif calculate == "f1":
    return f1


def TPR_FPR(dataset, gt_dataset, genome_size): #pass dataframe from variant caller and GT dataframe
    d_P=dataset["POS"].values
    g_P=gt_dataset["POS"].values
    matrix=np.zeros((2,2)) # matrix of 2 by 2
    matrix[1,1]=genome_size - len(gt_dataset) # True Negatives
    for i in d_P:
        if i in g_P:
            matrix[0,0]+=1 #True Positives
        elif i not in g_P:
            matrix[0,1]+=1 #False Positives
    for k in g_P:
      if k not in d_P:
        matrix[1,0]=+1 #False Negatives
    TPR=matrix[0,0]/(matrix[0,0]+matrix[1,0])
    FPR=matrix[0,1]/(matrix[1,1]+matrix[0,1])
    return TPR, FPR

def random_split_by_number(data,test_points):
 """return two matrices splitting the data at random
    Example:
    train, temp = random_split_by_number(data, 39)
    valid, test = random_split_by_number(temp, 20)
 """
 ranks = np.arange(data.shape[0])
 np.random.shuffle(ranks)
 train = data[ranks>=test_points,:]
 test = data[ranks<test_points,:]
 return train,test

def parse_vcf_mutect(fname, info_cols=None, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP': int, 'CIGAR': str}, nrows=1000)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)

    def parse_info(info_str):
        info_dict = {}
        for item in info_str.split(";"):
            parts = item.split("=")
            if len(parts) == 2:
                key, value = parts
            else:
                key = parts[0]
                value = ""
            info_dict[key] = value
        return info_dict

    vcf['INFO'] = vcf['INFO'].apply(parse_info)

    if info_cols is not None:
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    return vcf


### Define paths to files 
os.environ["ML_GT_FOLDER_PATH"] = "C:/Users/mmiskinyte/Documents/Python_ML/GT"
GT_folder_path = os.environ.get("ML_GT_FOLDER_PATH")

Mutect_path="C:/Users/mmiskinyte/Documents/Python_ML/Mutect2"
GATK_path="C:/Users/mmiskinyte/Documents/Python_ML/GATK"
Freebayes_path="C:/Users/mmiskinyte/Documents/Python_ML/freebayes/same_freq"



### Upload GT dataset

GT_file_path = os.path.join(GT_folder_path, "GT_832.csv")
filtered_GT_ML = pd.read_csv(GT_file_path)



### Upload Freebayes dataset

#assert os.path.exists(f'{Freebayes_path}/var05.vcf'), "The VCF file does not exist at the given path!"

#with open(f'{Freebayes_path}/var05.vcf', 'r') as file:
#    for _ in range(10):  # print first 10 lines
#        print(file.readline())


#header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
#vcf_df = pd.read_csv(f'{Freebayes_path}/var05.vcf', delimiter='\t', comment='#', names=header, nrows=10)
#print(vcf_df)



freebayes2 = parse_vcf(f'{Freebayes_path}/var05.vcf', info_cols=None, nrows=80000)


### Calculate confusion matrix

freebayes_confusion=confusion_matrix_m(freebayes2,filtered_GT_ML,genome_size=1267843)
#print(freebayes_confusion)

print((freebayes2))



### Join multiple vcf files for frequency of 0.5% by POS













### Print confusion matrix 

plot=plot_custom_confusion_matrix(freebayes_confusion, title="Freebayes 0.5%% variants")
plot

#test=parse_vcf("var05.vcf", info_cols=None,nrows=100)



# For saving local files
# GT_filtered2.to_csv('path_to_save_GT_filtered2.csv')

# For reading files from a folder
# folder_path = 'path_to_your_folder_with_VCF_files'
# ... [Rest of your loop to process files in the folder]


