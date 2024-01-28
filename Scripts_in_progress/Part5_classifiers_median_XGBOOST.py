# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 23:46:33 2023

@author: mmiskinyte
"""

import sklearn
sklearn_version = sklearn.__version__
sklearn_version

import sys
import os
import io
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
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

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from numpy import isnan
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier

from numpy import mean
#from numpy import stdfrom 
from numpy import std

from sklearn.svm import SVC
import itertools
from sklearn.feature_selection import SelectKBest, f_classif 
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif


from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score

import itertools
from sklearn.utils import class_weight
from sklearn.svm import SVC

from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight


import xgboost as xgb


#AUX

def plot_confusion_matrix_sklearn(confusion_matrix, title="Conf Matrix", save_path=None):
    """
    This function plots a custom 2x2 confusion matrix with specified colors by name and labels for TP, TN, FP, FN.
    It can also save the plot to a specified path.

    Args:
    confusion_matrix (list of lists): A 2x2 confusion matrix in the format [[TP, FP], [FN, TN]]
    title (str): Custom title for the confusion matrix (optional)
    save_path (str): Path to save the plot (optional)

    Returns:
    None
    """
    class_names = ['Positive', 'Negative']

    plt.imshow(confusion_matrix, interpolation='nearest', cmap='gray', vmin=0, vmax=1, aspect='auto')
    plt.title(title)

    tick_marks = [0, 1]
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    labels = ["TN", "FP", "FN", "TP"]
    for i in range(2):
        for j in range(2):
            if i == j:
                color = 'green'
            else:
                color = 'red'
            plt.text(j, i, str(confusion_matrix[i][j]), horizontalalignment="center", color=color)
            plt.text(j, i-0.2, labels[i*2+j], horizontalalignment="center")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if save_path:
        plt.savefig(save_path)
        plt.close()  # Close the plot after saving to prevent display in the notebook
    else:
        plt.show()





def metrics_sklearn(matrix, calculate=None):
    """
    Function to calculate metrics for different variant callers based on confusion matrix.
    If no second argument is given, returns all the metrics. If a particular metric is necessary,
    define the argument in the following way.
    Example:
    precision = metrics_m(matrix, "precision")
    """
    TP = matrix[1, 1]
    FP = matrix[0, 1]
    FN = matrix[1, 0]
    TN=matrix[0, 0]

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Check if both precision and recall are zero for F1 calculation
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * (recall * precision) / (recall + precision)

    accuracy = (TP + TN) / np.sum(matrix) if np.sum(matrix) > 0 else 0

    metrics_values = [precision, recall, accuracy, f1]

    if calculate is None:
        return metrics_values
    elif calculate == "precision":
        return precision
    elif calculate == "recall":
        return recall
    elif calculate == "accuracy":
        return accuracy
    elif calculate == "f1":
        return f1

def log_metrics(file_name, model_name, true_error, confusion_matrix, metrics, c_value):
    metric_names = ["Precision", "Recall", "Accuracy", "F1-Score"]
    with open(file_name, 'a') as file:
        file.write(f"Model: {model_name}, C: {c_value}\n")
        file.write(f"True Error: {np.round(true_error, 6)}\n")
        file.write("Confusion Matrix:\n")
        file.write(str(confusion_matrix) + "\n")
        file.write("Metrics:\n")
        for name, value in zip(metric_names, metrics):
            file.write(f"{name}: {value}\n")
        file.write("\n")

def log_metrics_gamma(file_name, model_name, true_error, confusion_matrix, metrics, c_value, gamma_value):
    metric_names = ["Precision", "Recall", "Accuracy", "F1-Score"]
    with open(file_name, 'a') as file:
        file.write(f"Model: {model_name}, C: {c_value}\n")
        file.write(f"Model: {model_name}, gamma: {gamma_value}\n")
        file.write(f"True Error: {np.round(true_error, 6)}\n")
        file.write("Confusion Matrix:\n")
        file.write(str(confusion_matrix) + "\n")
        file.write("Metrics:\n")
        for name, value in zip(metric_names, metrics):
            file.write(f"{name}: {value}\n")
        file.write("\n")




"""Load datasets"""


## Define paths to save files

final_for_ML="C:/Users/mmiskinyte/Documents/Python_ML/datasets_final_median"

log="C:/Users/mmiskinyte/Documents/Python_ML/datasets_final_median/XGBOOST"


y_train = "y_train_median_f.csv"
y_test = "y_test_median_f.csv"

x_train = "x_train_median_f.csv"
x_test = "x_test_median_f.csv"


# construct the full file paths

y_train_path = os.path.join(final_for_ML, y_train)
y_test_path = os.path.join(final_for_ML, y_test)
x_train_path = os.path.join(final_for_ML, x_train)
x_test_path = os.path.join(final_for_ML, x_test)


# Load the dataset directly into a DataFrame

y_train = pd.read_csv(y_train_path, delimiter=',', index_col=0)
y_test = pd.read_csv(y_test_path, delimiter=',', index_col=0)
x_train = pd.read_csv(x_train_path, delimiter=',', index_col=0)
x_test = pd.read_csv(x_test_path, delimiter=',', index_col=0)




"""XGBOOST classifier


If you are using XGBoost with decision trees as your base model, 
you don't need to worry about scaling or normalizing your features. 
Decision trees are not sensitive to the scale of the features.

"""

# Sanity checks
y_train.info(verbose=True) 
print(type(y_train))
print(y_train.head)

x_train.info(verbose=True) 
print(type(x_train))
print(x_train.head)

### Percentages match?

print(y_train.value_counts(normalize=True).values)
print(y_test.value_counts(normalize=True).values)

### Indices match?

print(x_train.index.equals(y_train.index)) #Indices match
print(x_test.index.equals(y_test.index)) #Indices match




"""XGB TO OPTIMISE"""

#kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} 
#gamma{‘scale’, ‘auto’} or float, default=’scale’ #scale might be better


# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced', 
    classes=np.unique(y_train.iloc[:,0]), 
    y=y_train.iloc[:,0]
)

# Create a dictionary of class weights
weight_dict = dict(zip(np.unique(y_train.iloc[:,0]), class_weights))

# Map each sample in the datasets to its corresponding weight
weights_train = y_train.iloc[:,0].map(weight_dict)
weights_test = y_test.iloc[:,0].map(weight_dict)

# Create DMatrix for each dataset with corresponding weights
D_train = xgb.DMatrix(x_train, label=y_train, weight=weights_train)
D_test = xgb.DMatrix(x_test, label=y_test, weight=weights_test)



# Define the parameters for the XGBoost classifier

# 0.01 and 0.3 for learning rate eta is typical. smaller eta value makes the model more robust to overfitting

# max_depth - higher depth will allow the model to learn relations very specific to a particular sample. 
#A max_depth of 3 means each tree in the ensemble can have at most 3 levels




neg, pos = np.bincount(y_train.iloc[:,0])
scale_pos_weight = neg / pos #specific to XGBoost designed to balance


""" This is super interesting!!!

For my highly imbalanced data, if I use binary logistic, it performs way worse then multi:softprob with 2 classes.

The multi:softprob objective produces a probability for each class, which could provide a richer set of information for certain types of datasets or problems, especially in cases where the decision boundary is not very distinct. 
In contrast, binary:logistic directly outputs predictions, which might be less nuanced in some scenarios.

Handling of Class Probabilities: multi:softprob might be handling the probabilities in a way that better suits your data, even for a binary classification task.
It's possible that the way probabilities are calculated and thresholds are applied in the multi:softprob setting aligns more effectively with my imbalanced dataset.

Note: check also for possible convergence issues? Numerical stability?

Also, multi:softprob could behave better with outliers.

"""





#This learns metrics_xgb [0.8125, 0.28888888888888886, 0.995092540661806, 0.4262295081967213]

param = {
    'eta': 0.01, 
    'max_depth': 50,  
    'objective': 'multi:softprob',  
    'num_class': 2,
    'scale_pos_weight': scale_pos_weight}  #num_class: This parameter is required only when objective is set to multi:softprob





# This does not learn and give metrics: metrics_xgb [0, 0.0, 0.9936904094223219, 0]
param2 = {
    'eta': 0.01, 
    'max_depth': 50,  
    'objective': 'binary:logistic',
    'scale_pos_weight': scale_pos_weight}  #num_class: This parameter is required only when objective is set to multi:softprob





steps = 20  # The number of training iterations


# Train 

model = xgb.train(param, D_train, steps)


# Predict the labels of the test set
preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

# Print the accuracy of the classifier
print(f"Accuracy: {accuracy_score(y_test, best_preds)}")



confusion_mat = confusion_matrix(y_test, best_preds)



metrics_xgb_A=metrics_sklearn(confusion_mat)
print("metrics_xgb",metrics_xgb_A)




# Train 
model = xgb.train(param2, D_train, steps)


# Predict the labels of the test set
preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

# Print the accuracy of the classifier
print(f"Accuracy: {accuracy_score(y_test, best_preds)}")



confusion_mat = confusion_matrix(y_test, best_preds)



metrics_xgb_A=metrics_sklearn(confusion_mat)
print("metrics_xgb",metrics_xgb_A)






param3 = {
    'eta': 0.01, 
    'max_depth': 50,  
    'objective': 'reg:tweedie',
    'scale_pos_weight': scale_pos_weight}  #num_class: This parameter is required only when objective is set to multi:softprob




steps = 50  # The number of training iterations

# Train the model
model = xgb.train(param3, D_train, steps)


# Predict the labels of the test set
preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

# Print the accuracy of the classifier
print(f"Accuracy: {accuracy_score(y_test, best_preds)}")



confusion_mat = confusion_matrix(y_test, best_preds)



metrics_xgb_A=metrics_sklearn(confusion_mat)
print("metrics_xgb",metrics_xgb_A)






""" Parameters to optimize:
    
    gamma [default=0, alias: min_split_loss] Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
    
    max_depth [default=6] 
    
    min_child_weight [default=1]  Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning.
    The larger min_child_weight is, the more conservative the algorithm will be.
    
    max_delta_step [default=0] Maximum delta step we allow each leaf output to be. 
    If the value is set to 0, it means there is no constraint. 
    If it is set to a positive value, it can help making the update step more conservative.
    Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
    Set it to value of 1-10 might help control the update.
    
    
    subsample [default=1] Subsample ratio of the training instances. 
    Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior 
    to growing trees. and this will prevent overfitting.
    Subsampling will occur once in every boosting iteration.range: (0,1]
                                                                    
                                                                    
    lambda [default=1, alias: reg_lambda] L2 regularization term on weights. Increasing this value will make model more conservative. 
    
    alpha [default=0, alias: reg_alpha] L1 regularization term on weights. Increasing this value will make model more conservative.



    scale_pos_weight [default=1]
    Control the balance of positive and negative weights, useful for unbalanced classes. A typical value to consider: sum(negative instances) / sum(positive instances). 

"""





steps = 50  # The number of training iterations


param = {
    'eta': 0.05, 
    'objective': 'multi:softprob',  
    'num_class': 2}  #num_class: This parameter is required only when objective is set to multi:softprob


max_depth = np.arange(3, 20, 1)
subsample_range = np.arange(0.5, 1.05, 0.1)
colsample_bytree_range = np.arange(0.5, 1.05, 0.1)

train_err_b = []
valid_err_b = []
folds = 5
kf = StratifiedKFold(n_splits=folds)


for depth in max_depth:
    tr_err = va_err = 0
    param['max_depth'] = depth  # Set max_depth in param

    for tr_ix, va_ix in kf.split(x_train, y_train):
        D_train_fold = xgb.DMatrix(x_train.iloc[tr_ix], label=y_train.iloc[tr_ix])
        D_valid_fold = xgb.DMatrix(x_train.iloc[va_ix], label=y_train.iloc[va_ix])

        model = xgb.train(params=param, dtrain=D_train_fold, num_boost_round=steps)

        
        #model.predict is a 2D array where each row corresponds to a sample, 
        #and each column corresponds to the predicted probability of that sample belonging to a 
        #particular class. 
        #This is different from binary classification where the output is typically a 1D array of labels or values.
        #multi:softprob returns probabilities, to get the actual class predictions, 
        #I need to select the class with the highest probability for each sample np.argmax.
        #
        
        
        y_t_pred = np.argmax(model.predict(D_train_fold), axis=1) #maximum values along the axis 1 (columns), effectively converting probability distributions to class labels.
        y_v_pred = np.argmax(model.predict(D_valid_fold), axis=1)

        xgb_t = 1 - balanced_accuracy_score(y_train.iloc[tr_ix], y_t_pred)
        tr_err += xgb_t
        
        xgb_v = 1 - balanced_accuracy_score(y_train.iloc[va_ix], y_v_pred)
        va_err += xgb_v
    
    train_err_b.append(tr_err / folds)
    valid_err_b.append(va_err / folds)
    print(f"{depth}: {tr_err / folds}, {va_err / folds}")



best_depth = max_depth[np.argmin(valid_err_b)]
print("Best max_depth:", best_depth)


#max_depth = np.arange(6, 60, 2)
plt.figure(figsize=(12, 8))
plt.plot(max_depth, train_err_b, c='b', label = 'Training Error')
plt.plot(max_depth, valid_err_b, c='r', label = 'Validation Error')
#plt.axis([6, 60, 0.0, 0.5])
plt.title('Training and validation errors for different depth (xgb)')
plt.xlabel('Max depth')
plt.ylabel('Error')
plt.legend(loc='lower right',frameon=False)
save_path = "C:/Users/mmiskinyte/Documents/Python_ML/datasets_final_median/XGBOOST/xgb_plot.png"
plt.savefig(save_path)

# Close the plot to prevent it from displaying in the notebook
plt.close()





# Retrain the model with best max_depth
param['max_depth'] = best_depth
final_model = xgb.train(param, D_train, steps)

# Evaluate on test set
preds = final_model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

# Calculate test error and other metrics
xgb_true_error = 1 - balanced_accuracy_score(y_test, best_preds)
print('xgb true error:', xgb_true_error)

# Confusion Matrix



confusion_path="C:/Users/mmiskinyte/Documents/Python_ML/datasets_final_median/XGBOOST/xgb_v_confusion.png"
confusion_mat = confusion_matrix(y_test, best_preds)
plot_confusion_matrix_sklearn(confusion_mat, title="Confusion matrix(xgb)", save_path=confusion_path)



# Metrics

#precision, recall, accuracy, f1
metrics_xgb=metrics_sklearn(confusion_mat)
print("metrics_xgb",metrics_xgb)



#log
log_metrics(f"{log}/my_log.txt", "xgbC", xgb_true_error, confusion_mat, metrics_xgb, best_depth)












# new




max_depth_range = np.arange(3, 20, 1)
subsample_range = np.arange(0.5, 1.05, 0.1)
colsample_bytree_range = np.arange(0.5, 1.05, 0.1)




results = []

# Grid search
for depth in max_depth_range:
    for subsample in subsample_range:
        for colsample_bytree in colsample_bytree_range:
            param['max_depth'] = depth
            param['subsample'] = subsample
            param['colsample_bytree'] = colsample_bytree

            tr_err = va_err = 0
            for tr_ix, va_ix in kf.split(x_train, y_train):
                D_train_fold = xgb.DMatrix(x_train.iloc[tr_ix], label=y_train.iloc[tr_ix])
                D_valid_fold = xgb.DMatrix(x_train.iloc[va_ix], label=y_train.iloc[va_ix])

                model = xgb.train(params=param, dtrain=D_train_fold, num_boost_round=steps)

                y_t_pred = np.argmax(model.predict(D_train_fold), axis=1)
                y_v_pred = np.argmax(model.predict(D_valid_fold), axis=1)

                xgb_t = 1 - balanced_accuracy_score(y_train.iloc[tr_ix], y_t_pred)
                xgb_v = 1 - balanced_accuracy_score(y_train.iloc[va_ix], y_v_pred)

                tr_err += xgb_t
                va_err += xgb_v
            
            # Average error across folds
            avg_tr_err = tr_err / folds
            avg_va_err = va_err / folds

            results.append((depth, subsample, colsample_bytree, avg_tr_err, avg_va_err))
            print(f"Depth: {depth}, Subsample: {subsample}, Colsample_bytree: {colsample_bytree}, Tr Error: {avg_tr_err}, Val Error: {avg_va_err}")

# Find the best parameters
best_params = min(results, key=lambda x: x[4])
print("Best Parameters:", best_params)
















