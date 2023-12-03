# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 14:58:34 2023

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
import functools as ft
from itertools import combinations

import re
### AUX functions

def parse_vcf(fname, info_cols=None, parse_all_info=False, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    parse_all_info: If True, parse all INFO columns into separate columns.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP':int,'CIGAR':str,}, nrows=1000, parse_all_info=True)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    # create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    
    if parse_all_info:
        # Parse all INFO columns into separate columns
        for field, value in vcf['INFO'][0].items():
            vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
    elif info_cols is not None:
        # Parse specified INFO columns into separate columns
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    vcf.drop(columns=['INFO'], inplace=True)
    
    return vcf





def parse_vcf_mutect(fname, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary, and each key-value pair is expanded into distinct columns.
    nrows: how many rows to read from the start of the header.
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)

    # Function to parse the INFO field into a dictionary
    def parse_info(info_str):
        info_dict = {}
        for item in info_str.split(";"):
            parts = item.split("=")
            if len(parts) == 2:
                key, value = parts
                info_dict[key] = value
        return info_dict

    # Apply parse_info function to each row in the INFO column
    vcf['INFO'] = vcf['INFO'].apply(parse_info)

    # Create a list of multi-valued key columns to drop later
    multi_key_columns = []

    # Parse columns with single keys
    single_keys = set()
    for info_dict in vcf['INFO']:
        single_keys.update(info_dict.keys())

    for key in single_keys:
        vcf[key] = vcf['INFO'].apply(lambda x: x.get(key, None))
        vcf[key] = pd.to_numeric(vcf[key], errors='ignore')

    # Create separate columns for multi-valued keys (e.g., PV4)
    for key in vcf['INFO'].apply(lambda x: list(x.keys())).explode().unique():
        if vcf[key].apply(lambda x: isinstance(x, str) and ',' in x).any():
            values = vcf[key].str.split(',', expand=True)
            num_columns = values.shape[1]
            for i in range(1, num_columns + 1):
                new_key = f'{key}_{i}'
                vcf[new_key] = values[i - 1]
                vcf[new_key] = pd.to_numeric(vcf[new_key], errors='ignore')
            multi_key_columns.append(key)  # Add the multi-valued key to the list

    # Drop the original multi-valued key columns
    vcf.drop(columns=multi_key_columns, inplace=True)

    # Drop the original INFO column if no longer needed
    vcf.drop(columns=['INFO'], inplace=True)

    return vcf



def parse_vcf_varscan(fname, info_cols=None, parse_all_info=False, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    parse_all_info: If True, parse all INFO columns into separate columns.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP':int,'CIGAR':str,}, nrows=1000, parse_all_info=True)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    # create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    
    if parse_all_info:
        # Parse all INFO columns into separate columns
        for field, value in vcf['INFO'][0].items():
            vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
    elif info_cols is not None:
        # Parse specified INFO columns into separate columns
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    vcf.drop(columns=['INFO'], inplace=True)
    
    return vcf



def parse_vcf_varscan2(fname, info_cols=None, parse_all_info=False, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    The GT column is split into separate columns based on the ':' delimiter.
    nrows: how many rows to read from the start of the header.
    parse_all_info: If True, parse all INFO columns into separate columns.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP': int, 'CIGAR': str}, nrows=1000, parse_all_info=True)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    
    # Create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    
    if parse_all_info:
        # Parse all INFO columns into separate columns
        for field, value in vcf['INFO'][0].items():
            vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
    elif info_cols is not None:
        # Parse specified INFO columns into separate columns
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    
    # Split the 'GT' column into separate columns
    gt_columns = vcf['FORMAT'].str.split(':').tolist()
    gt_values = vcf['GT'].str.split(':').tolist()
    
    #print("Length of gt_columns:", len(gt_columns))
   # print("Length of gt_values:", len(gt_values))
    
    for i, col in enumerate(gt_columns):
        vcf[col] = [val[i] for val in gt_values]
    
    vcf.drop(columns=['INFO', 'GT'], inplace=True)
    
    return vcf




def parse_vcf_varscan3(fname, info_cols=None, parse_all_info=False, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    parse_all_info: If True, parse all INFO columns into separate columns.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP':int,'CIGAR':str,}, nrows=1000, parse_all_info=True)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    
    # Create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    
    if parse_all_info:
        # Parse all INFO columns into separate columns
        for field, value in vcf['INFO'][0].items():
            vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
    elif info_cols is not None:
        # Parse specified INFO columns into separate columns
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    

    
    # Split FORMAT and GT columns by ":" and create new columns for each element
    vcf['FORMAT'] = vcf['FORMAT'].str.split(":")
    vcf['GT'] = vcf['GT'].str.split(":")
    
    vcf = pd.concat([vcf, vcf['GT'].apply(lambda x: pd.Series(x))], axis=1)
    vcf.drop(columns=['INFO','GT','FORMAT'], inplace=True)
    
    return vcf




def parse_vcf_varscan(fname, info_cols=None, parse_all_info=False, nrows=None):
    """Parse a VCF file into a dataframe.
    The INFO column is parsed into a dictionary with specified dtype in distinct column.
    nrows: how many rows to read from the start of the header.
    parse_all_info: If True, parse all INFO columns into separate columns.
    Example:
    vcf_df_GT = parse_vcf('test.vcf', info_cols={'DP':int,'CIGAR':str,}, nrows=1000, parse_all_info=True)
    """
    header = "CHROM POS ID REF ALT QUAL FILTER INFO FORMAT GT".split()
    vcf = pd.read_csv(
        fname, delimiter='\t', comment='#', names=header, nrows=nrows)
    
    # Create a dictionary out of INFO field
    vcf['INFO'] = vcf['INFO'].str.split(";").apply(lambda x: dict([y.split("=") for y in x]))
    
    if parse_all_info:
        # Parse all INFO columns into separate columns
        for field, value in vcf['INFO'][0].items():
            vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
    elif info_cols is not None:
        # Parse specified INFO columns into separate columns
        for field, dtype in info_cols.items():
            try:
                vcf[field] = vcf['INFO'].apply(lambda x: x.get(field, None))
                vcf[field] = vcf[field].astype(dtype)
            except:
                pass
    
    # Parse FORMAT into column names
    format_cols = vcf['FORMAT'].str.split(":").iloc[0]
    
    # Split GT columns by ":" and create new columns with names from FORMAT
    gt_data = vcf['GT'].str.split(":", expand=True)
    gt_data.columns = format_cols
    
    vcf = pd.concat([vcf, gt_data], axis=1)
    vcf.drop(columns=['INFO','GT','FORMAT'], inplace=True)
    
    return vcf





### Define paths to files and load files
os.environ["ML_GT_FOLDER_PATH"] = "C:/Users/mmiskinyte/Documents/Python_ML/GT"
GT_folder_path = os.environ.get("ML_GT_FOLDER_PATH")
GT_file_path = os.path.join(GT_folder_path, "GT_filtered_for_ML.csv")
filtered_GT_ML = pd.read_csv(GT_file_path, delimiter=';')

variants_path="C:/Users/mmiskinyte/Documents/Python_ML/all_sites"

freebayes05 = parse_vcf3(f'{variants_path}/freebayes05.vcf', info_cols=None, parse_all_info=True, nrows=1300000)
bcftools05=parse_vcf_mutect8(f'{variants_path}/bcftools05.vcf', nrows=1300000)
varscan05=parse_vcf_varscan4(f'{variants_path}/varscanlf05.vcf', info_cols=None, parse_all_info=True, nrows=1300000)





### Save new files

#OK
#bcftools05t1 = parse_vcf_mutect4(f'{variants_path}/bcftools05.vcf', info_cols=info_cols_min, nrows=100)

#OK
#test2=parse_vcf_mutect6(f'{variants_path}/bcftools05.vcf', nrows=4000)

#OK
#test3=parse_vcf_mutect7(f'{variants_path}/bcftools05.vcf', nrows=4000)


#test2.to_csv(f'{variants_path}/test2.csv')
#test3.to_csv(f'{variants_path}/test3.csv')

bcftools05.to_csv(f'{variants_path}/bcftools05.csv')
freebayes05.to_csv(f'{variants_path}/freebayes05.csv')
varscan05.to_csv(f'{variants_path}/varscan05.csv')




### Now join all datasets into one dataset


####REMOVE THIS PART FROM HERE LATER

# dfs = [bcftools05, freebayes05, varscan05]

# for df in [bcftools05, freebayes05, varscan05]:
#     for col in df.columns:
#         # Check if any entry in the column is a dictionary
#         if any(isinstance(x, dict) for x in df[col]):
#             df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)

# # Merge the DataFrames on 'POS'
# df_f = ft.reduce(lambda left, right: pd.merge(left, right, on='POS'), dfs)

# df_final2 = ft.reduce(lambda left, right: pd.merge(left, right, on='POS', how='left'), dfs)


# # After ensuring no column contains dicts, filter out the columns that have the same value in all rows
# df_f = df_f.loc[:, df_f.nunique() != 1]
# df_final2 = df_final2.loc[:, df_f.nunique() != 1]


# #df_f.to_csv(f'{variants_path}/df_f1.csv')


# duplicated_column_pairs = []

# Iterate over each combination of columns
# for col1, col2 in combinations(df_f.columns, 2):
#     if df_f[col1].equals(df_f[col2]):
#         # If the columns have the same values, add the pair of column names to the list
#         duplicated_column_pairs.append((col1, col2))

# # Check if we found any duplicated columns
# if duplicated_column_pairs:
#     print("Duplicated columns found:")
#     for col1, col2 in duplicated_column_pairs:
#         print(f"Column: {col1} is a duplicate of Column: {col2}")
# else:
#     print("No duplicated columns found.")



### UNTIL HERE


def preprocess_dataframes(dfs):
    # Convert any dictionary entries to strings across all columns and DataFrames
    for df in dfs:
        for col in df.columns:
            # Check if any entry in the column is a dictionary
            if any(isinstance(x, dict) for x in df[col]):
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, dict) else x)
    
    # Merge the DataFrames on 'POS' based on a first dataframe in a list of dataframes
    df_f = ft.reduce(lambda left, right: pd.merge(left, right, on='POS', how='left'), dfs)
    
    # Remove columns with the same value across all rows
    df_f = df_f.loc[:, df_f.nunique() != 1]
    
    # Function to find duplicated columns and keep only one of each duplicate
    def remove_duplicate_columns(df):
        # While there are duplicates, keep removing them
        while True:
            # Find duplicated columns
            duplicated_cols = []
            for col1, col2 in combinations(df.columns, 2):
                if col1 != col2 and df[col1].equals(df[col2]):
                    duplicated_cols.append(col2)
            # If no duplicates, break
            if not duplicated_cols:
                break
            # Remove duplicated columns
            df = df.drop(columns=duplicated_cols)
        return df
    
    # Apply the function to remove duplicate columns
    df_f = remove_duplicate_columns(df_f)
    
    return df_f



dfs = [bcftools05, freebayes05, varscan05]
preprocessed_df = preprocess_dataframes(dfs)

preprocessed_df.to_csv(f'{variants_path}/preprocessed_df.csv')









