# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:16:00 2023

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




### Define paths to files and load files
os.environ["DATASET_FOLDER_PATH"] = "C:/Users/mmiskinyte/Documents/Python_ML/all_sites"
df_folder_path = os.environ.get("DATASET_FOLDER_PATH")
df_file_path = os.path.join(df_folder_path, "preprocessed_df.csv")
df_1 = pd.read_csv(df_file_path, delimiter=',', index_col=0)

print(type(df_1))

print("Data Types of All Columns:")
print(df_1.dtypes)


missing_values_percent = df_1.isna().mean() * 100


# Print the percentage of missing values for each column
print("Percentage of missing values per column:")
print(missing_values_percent)


threshold = 99.99

percent_missing = df_1.isnull().mean() * 100

columns_to_keep = percent_missing[percent_missing < threshold].index
df_2 = df_1[columns_to_keep]



# columns_to_remove = ['REF', 'REF_y']

# # # Remove the specified columns
# df_3 = df_3.drop(columns=columns_to_remove)

# for column in df_filtered.columns:
#         if 'REF' in column:
#             non_empty_count = df_filtered[column].dropna().shape[0]
#             print(f"Column '{column}' has {non_empty_count} non-empty rows.")




def data_imputation(df, threshold_missing_data,imputation_methods=None):
    
    
    '''
    This function is required for data clean up and imputation.
    First, it deletes columns that has a percentage of missing data that is above user defined threshold.
    Second, it 
    
    
    Example of how to use function:
        imputation_methods = {'column1': 'mean', 'column2': 42, 'column3': 'median'}
        df_new=remove_bad_columns(df, imputation_methods, threshold_missing_data=99.99)
    
    
    '''
    threshold=threshold_missing_data
    max_count = 0
    max_column = None
    
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df_test = df.copy()
    
    # Remove all columns that has more than 99.99% values missing in rows
    
    percent_missing = df_test.isnull().mean() * 100

    columns_to_keep = percent_missing[percent_missing < threshold].index
    df_test = df_test[columns_to_keep]
    
    
    # Check which REF column (several were generated due to join by POS from different variant callers) has more rows and keep only that column
    ref_columns = [col for col in df_test.columns if 'REF' in col] 
    if len(ref_columns) > 1:
        for column in ref_columns: 
            non_empty_count = df_test[column].dropna().shape[0] 
            if non_empty_count > max_count:
                max_count = non_empty_count
                max_column = column

        if max_column:
            print(f"Column '{max_column}' with highest number of non-empty rows: {max_count} is kept")
            columns_to_drop = [col for col in ref_columns if col != max_column]
            df_test.drop(columns=columns_to_drop, inplace=True)
        
    elif len(ref_columns) == 1:
        print(f"Only one REF column '{ref_columns[0]}' found in the DataFrame. Keeping it.")
    else: 
        print("No 'REF' columns found.")
    
        
        
    # Now replace empty rows of chosen columns either with max value, mean value or a specific number
    if not imputation_methods:
        print("You did not impute any missing values.")
        return df_test

    for column, impute_info in imputation_methods.items():
        if column in df_test.columns:
            impute_method = impute_info['method']
            impute_type = impute_info['type']

            if impute_type == 'string':
                # String imputation
                df_test[column].fillna(impute_method, inplace=True)
            elif impute_type in ['float', 'numeric']:
                # Convert column to numeric, if possible
                df_test[column] = pd.to_numeric(df_test[column], errors='coerce')
                
                # Numeric imputation methods
                if impute_method == 'max':
                    df_test[column].fillna(df_test[column].max(), inplace=True)
                elif impute_method == 'min':
                    df_test[column].fillna(df_test[column].min(), inplace=True)
                elif impute_method == 'mean':
                    df_test[column].fillna(df_test[column].mean(), inplace=True)
                elif impute_method == 'median':
                    df_test[column].fillna(df_test[column].median(), inplace=True)
                else:  # Assuming impute_method is a specific number
                    df_test[column].fillna(float(impute_method), inplace=True)
            else:
                print(f"Imputation type '{impute_type}' not recognized for column '{column}'")
        else:
            print(f"Column '{column}' not found in the DataFrame.")
    
        
        # Check for remaining missing data
    missing_after_imputation = df_test.isnull().mean() * 100
    missing_columns = missing_after_imputation[missing_after_imputation > 0] 
    if missing_columns.empty: 
        print("All data successfully imputed.")
    else: 
        print("Columns with remaining missing data:") 
        for column, missing_percent in missing_columns.items(): 
            formatted_percent = "{:.2f}%".format(missing_percent)
            print(f"{column}: {formatted_percent} missing data")        
         
    return df_test


#imputation_methods = {'PVAL': 'max'}

imputation_methods = {
    'PVAL': {'method': '1.0', 'type': 'float'},
    'PV4_1': {'method': '1.0', 'type': 'float'},
    'PV4_2': {'method': 'max', 'type': 'numeric'},
    'PV4_3': {'method': 'max', 'type': 'numeric'},
    'PV4_4': {'method': 'max', 'type': 'numeric'}
}

imputed_test=data_imputation(df_1,  threshold_missing_data=99, imputation_methods=imputation_methods)

print("Data Types of All Columns:")
print(imputed_test.dtypes)

imputation_path="C:/Users/mmiskinyte/Documents/Python_ML/imputation_tests"
#imputed_test.to_csv(f'{imputation_path}/imputed_test.csv')

# Example usage
# df = pd.read_your_dataframe_here()  # Load your DataFrame
# imputation_methods = {'PVAL': 'max', 'PV4_1': 'max', 'PV4_2': 'max','PV4_3': 'max','PV4_4': 'max'}
# df = impute_values(df, imputation_methods)





# Example usage
# df = pd.read_your_dataframe_here()  # Replace with your dataframe loading code
# df = update_df_with_ref_column_with_most_non_empty_rows(df)

### A) Re-write script too search for string *REF* in the columns, 
#find all columns based on regex and then keep only column that has most non missing values

### B) If in columns that has *ALT* in col name, is null, fill in with string REF 

### C) Pval imputation, for columns with names PVAL, PV_1, PV_2, PV_3, PV_4 if value is absent, replace 1




#fill in with largest number for missing values

### check what is MQBZ, MQSBZ and VDB - they seem important



info_fields = {
    "POS": "Position of the variant on the reference chromosome.",
    "REF_x": "Reference base(s) at the position of the variant.",
    "ALT_x": "Alternate base(s) at the position of the variant, different from the reference base(s).",
    "QUAL_x": "Quality score for the assertion made in ALT. Higher scores indicate higher confidence.",
    "GT_x": "Genotype, indicating the genetic makeup at the variant position.",
    "IMF": "Inbreeding/Informativeness for Missingness, measures informativeness of a locus for inbreeding.",
    "SGB": "Statistical support for base quality scores being biased/unbiased.",
    "MQBZ": "Z-score from the Mann-Whitney U test of Mapping Quality vs. Base Quality.",
    "MQSBZ": "Z-score from the Mann-Whitney U test of Mapping Quality vs. Segregation (strand bias).",
    "IDV": "Number of indel-supporting reads.",
    "VDB": "Variant Distance Bias for filtering out false positive calls.",
    "DP_x": "Read depth at this position for this sample.",
    "AF1": "Estimated allele frequency of the variant.",
    "RPBZ": "Z-score from the Mann-Whitney U test of Read Position Bias.",
    "MQ": "Mapping quality.",
    "AC1": "Allele count in genotypes for the first allele listed in ALT.",
    "BQBZ": "Z-score from the Mann-Whitney U test of Base Quality Bias.",
    "MQ0F": "Fraction of MQ0 reads (reads with mapping quality zero).",
    "DP4_1": "Count of reference-supporting reads on the forward strand.",
    "DP4_2": "Count of reference-supporting reads on the reverse strand.",
    "DP4_3": "Count of alternate-supporting reads on the forward strand.",
    "DP4_4": "Count of alternate-supporting reads on the reverse strand.",
    "PV4_1": "P-value for strand bias.",
    "PV4_2": "P-value for base quality bias.",
    "PV4_3": "P-value for map quality bias.",
    "PV4_4": "P-value for tail distance bias.",
    "REF_y": "Similar to REF_x, possibly representing another sample or sub-population.",
    "ALT_y": "Similar to ALT_x, possibly representing another sample or sub-population.",
    "QUAL_y": "Similar to QUAL_x, possibly representing another sample or sub-population.",
    "GT_y": "Similar to GT_x, possibly representing another sample or sub-population.",
    "DP_y": "Similar to DP_x, representing another sample or sub-population.",
    "DPB": "Depth per base.",
    "EPPR": "End Placement Probability for reference alleles.",
    "MQMR": "Mapping Quality Mean Rank.",
    "NUMALT": "Number of alternate alleles.",
    "ODDS": "Odds ratio for the likelihood of the variant being present.",
    "PAIREDR": "Proportion of reads supporting the reference allele and properly paired.",
    "PQR": "Quality ratio for reference allele.",
    "PRO": "Proportion of reads supporting the reference allele.",
    "QR": "Quality of reads supporting the reference allele.",
    "RO": "Reads count supporting the reference allele.",
    "RPPR": "Read Placement Probability for reference alleles.",
    "ADP": "Average depth of coverage per allele.",
    "GQ": "Genotype Quality.",
    "RD": "Reads Depth for the reference allele.",
    "AD": "Allele Depth, number of reads supporting each allele.",
    "FREQ": "Frequency of this allele in the population.",
    "PVAL": "P-value for statistical tests.",
    "RBQ": "Base Quality for reference alleles.",
    "ABQ": "Base Quality for alternate alleles.",
    "RDF": "Depth of forward reads for reference allele.",
    "RDR": "Depth of reverse reads for reference allele.",
    "ADF": "Depth of forward reads for alternate allele.",
    "ADR": "Depth of reverse reads for alternate allele."
}

print(info_fields.keys())


# duplicated_column_pairs = []

# #Iterate over each combination of columns
# for col1, col2 in combinations(df_filtered.columns, 2):
#     if df_filtered[col1].equals(df_filtered[col2]):
#         # If the columns have the same values, add the pair of column names to the list
#         duplicated_column_pairs.append((col1, col2))

# # Check if we found any duplicated columns
# if duplicated_column_pairs:
#     print("Duplicated columns found:")
#     for col1, col2 in duplicated_column_pairs:
#         print(f"Column: {col1} is a duplicate of Column: {col2}")
# else:
#     print("No duplicated columns found.")







