# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:25:32 2022

@author: dbrody
"""

import numpy as np
import pandas as pd
from gain import gain
from scipy.stats import kstest

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from hyperopt import tpe
from hyperopt import STATUS_OK
from hyperopt import Trials
from hyperopt import hp
from hyperopt import fmin
from pprint import pprint
from matplotlib import pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler


from scipy.stats import ks_2samp, anderson_ksamp

import sys

N_FOLDS = 10
MAX_EVALS = 10



def unpack(x):
    if x:
        return x[0]
    return np.nan

def objective(params, n_folds = N_FOLDS):
    
    '''
    
    computes GAIN over different hint rates and alphas as a term of cross validation.
    Validation sets are created using indices generated by KFold.
    
    INPUTS:
        
    X_test : (array)  test feature set
    y_test : (array) test groud truth set
    hint_rate_vals  : (array) array of possible hint rate values
    alpha_vals  : (array) array of possible alpha values
    indices : (array) indices for validation set
    
    OUTPUTS:
        
    cross_val_1 : array containg hint rate, alpha, and nrmse in their respective columns
    for all possible combinations of hint rate and alpha in hint_rate_vals and alpha_vals
    
    '''
    
    #initializing cross_val_method which will be used to store hint rates and alphas
    #also initializing nrmses which will store nrmses for the respective hint rate and alpha
    
    nrmses = []
   
    cross_val_data= params['X_test']
   
    cross_val_ground_truth = params['y_test']
    
    #stratified k-fold
    
    skf = KFold(n_splits = 2)
                    
    gain_params = {
              'alpha' : params['alpha'],
              'hint_rate' : params['hint_rate'],
              'iterations': 1000,
              'mode' : params['mode'],
              'batch_size':20
              
              }
   
    

    for train_index, test_index in skf.split(cross_val_data, cross_val_ground_truth):
        
       
        data_split = cross_val_data[train_index]
        ground_truth_split = cross_val_ground_truth[train_index]
        
        print(np.shape(data_split))
    
    
        imputed_data_x = gain(np.array(data_split), gain_params)
    
        flat_imputed_data = np.reshape(imputed_data_x.T,(-1,1))
        flat_toy_data = np.reshape(data_split.T,(-1,1))
        flat_ground_truth = np.reshape(ground_truth_split.T,(-1,1))
    
        squared_error = []
        #total_col_imputed_data = []
        
        for col_idx in range(np.shape(ground_truth_split)[1]):
            
            
            
            col_length = np.shape(ground_truth_split)[0]
            col_indices = range(col_idx*col_length,(col_idx+1)*col_length)
            
            col_ground_truth = flat_ground_truth[col_indices]
      
            col_imputed_data = flat_imputed_data[col_indices]
            col_flat_toy_data = flat_toy_data[col_indices]
            
            max_col = np.max(col_ground_truth)
            min_col = np.min(col_ground_truth)
            
            col_squared_error = []
            #col_imputed_data = []
            
    
            for idx,j in enumerate(col_imputed_data):
                if(col_flat_toy_data[idx] != col_flat_toy_data[idx]):
                    col_squared_error.append((col_imputed_data[idx] - col_ground_truth[idx])**2)
                    #col_imputed_data.append(flat_imputed_data[idx])
            norm_col_nrmse = np.sqrt(np.nanmean(col_squared_error)) / (max_col-min_col)
            squared_error.append(norm_col_nrmse)
            #total_col_imputed_data.append([col_imputed_data, max_col, min_col])
                            
                    
        nrmse = np.nanmean(squared_error)
        nrmses.append(nrmse)
    loss = np.mean(nrmses)
    print(f'alpha : {params["alpha"]},  hint_rate : {params["hint_rate"]},   nrmse : {loss}')
    
    #maybe add 'imputed_data': total_col_imputed_data
                    
                   
    
    return {'loss': loss,'status': STATUS_OK}


def evaluate_gain(method, X_test, y_test, X_train):
    #splitting train and test to create a v
        gain_mode = "Normal"
        
        if(method == 'MIM-GAIN'):
            gain_mode = 'MIM'
        
        space = {
            'alpha' : hp.choice('alpha',[0.1, 0.5,1,2,10]), #choices used in paper (gain)
            'hint_rate' : hp.uniform('hint_rate',0,1),
            'mode' : gain_mode,
            'X_test' : X_test,
            'y_test' : y_test
            
            
            }
        
        tpe_algorithm = tpe.suggest
        bayes_trials = Trials()
        best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = bayes_trials)
        
        print(best)
        
        
        gain_parameters = {'batch_size': 100,
                     'hint_rate': best['hint_rate'],
                     'alpha': best['alpha'],
                     'iterations': 10000,
                     'mode' : gain_mode}
        
        imputed_data_x = gain(np.array(X_train), gain_parameters)
        
        return imputed_data_x
    
def evaluate_mean(X_train):
    data = pd.DataFrame(X_train)
    imputed_data_x = data.copy() 
    for col in data.columns:
        imputed_data_x[col].fillna(data[col].mean(), inplace = True)
    imputed_data_x = imputed_data_x.values 
    
    return imputed_data_x

def evaluate_median(X_train):
    data = pd.DataFrame(X_train)
    imputed_data_x = data.copy()
    for col in data.columns:
            
        imputed_data_x[col].fillna(data[col].median(), inplace = True)
            
    imputed_data_x = imputed_data_x.values
    
    return imputed_data_x


def evaluate(X_train,
             X_test,
             y_train,
             y_test,
             missing_rate,
             missingness,
             trial_num,
             dataset_name,
             method = "GAIN", 
             hint_rate_vals = None,
             alpha_vals = None,
             target = None,
             method_by_col = None):
    
    '''
    
    evaluates an algorith/method on the data and computes nrmse and auc based on predictions and ground truth
    
    INPUTS:
        
    data : (DataFrame) data to run the algorithm/method on
    method : (string) algorithm/method being tested
    ground_truth : (DataFrame) ground truth for data
    hint_rate_vals  : (array) array of possible hint rate values
    alpha_vals  : (array) array of possible alpha values
    
    OUTPUTS:
        
    nrmse : nrmse performance on dataset
    auc : auc performance on dataset
    validation : array containing the output of cross_val_gain for each split in KFold
    
    
    '''
    
    imputed_data_x = 0
    validation = []
    trials_df = []
    
    #allowing for if method  = GAIN
    
    if(method == 'GAIN' or method == 'MIM-GAIN'):
        imputed_data_x = evaluate_gain(method, X_test, y_test, X_train)
    
    elif(method == 'mean'):
        imputed_data_x = evaluate_mean(X_train)
        
        
    elif(method == 'median'):
        imputed_data_x = evaluate_median(X_train)
        
    elif(method == 'combination'):
        imputed_data_gain = evaluate_gain('GAIN', X_test, y_test, X_train)
        
        imputed_data_median = evaluate_median(X_train)
       
        
        
        col_length = np.shape(imputed_data_median)[1]
        imputed_data_x = np.zeros(np.shape(imputed_data_median))
        for i in range(col_length):
            if(method_by_col[i]== 0):
                imputed_data_x[:,i] = imputed_data_gain[:,i]
            elif(method_by_col[i]== 1):
                imputed_data_x[:,i] = imputed_data_median[:,i]
            
            
                
                
        
    sns.set(rc={'figure.figsize':(13,9)})
        
    fig, axes = plt.subplots(1, 3)
    
    corr_est = pd.DataFrame((imputed_data_x)).corr()
    corr_real = pd.DataFrame((y_train)).corr()
    corr_diff = abs(corr_est-corr_real)
    
    mse_corr = np.round_(np.mean((corr_est.values-corr_real.values)**2),3)
    
    sns.heatmap(corr_real, ax = axes[0])
    axes[0].set_title("ground truth", fontsize = 20)
    
    sns.heatmap(corr_est, ax = axes[1])
    axes[1].set_title(f"{method}", fontsize = 20)
    
    sns.heatmap(corr_diff)
    axes[2].set_title(f"difference (mse : {mse_corr})", fontsize = 20)
    
    fig.suptitle(f'Correlation Differences b/w Estimate and Ground Truth : Missing Rate = {missing_rate}, Missingness = {missingness}', fontsize = 20)
    
    path = r'C:\Users\potat\Downloads\mimic-lstm-master\mimic-lstm-master\toy_dataset_experiments'
    
   
    if not(os.path.isdir(path)):
        os.mkdir(path)
    
    path = r'C:\Users\potat\Downloads\mimic-lstm-master\mimic-lstm-master\toy_dataset_experiments\corr'
    
    if not(os.path.isdir(path)):
        os.mkdir(path)
        
    path = os.path.join(path, dataset_name)
    if not(os.path.isdir(path)):
        os.mkdir(path)
    
    path = os.path.join(path, str(missingness))
    if not(os.path.isdir(path)):
        os.mkdir(path)
    
    path = os.path.join(path, str(missing_rate))
    if not(os.path.isdir(path)):
        os.mkdir(path)
    
    fig.savefig(os.path.join(path,f'{method}_trial_{trial_num}.jpg'))
    
   
        
    flat_imputed_data = np.reshape(imputed_data_x.T,(-1,1))
    flat_toy_data = np.reshape(X_train.T,(-1,1))
    flat_ground_truth = np.reshape(y_train.T,(-1,1))
    
    

    squared_error = []
    total_col_imputed_data = []
    
    
    for col_idx in range(np.shape(X_train)[1]):
        col_length = np.shape(X_train)[0]
        col_indices = range(col_idx*col_length,(col_idx+1)*col_length)
        
        col_ground_truth = flat_ground_truth[col_indices]
      
        col_imputed_data = flat_imputed_data[col_indices]
        col_flat_toy_data = flat_toy_data[col_indices]
            
        max_col = np.max(col_ground_truth)
        min_col = np.min(col_ground_truth)
            
        col_squared_error = []
        col_replacements = []
        
    
    
        for idx,j in enumerate(col_imputed_data):
            if(col_flat_toy_data[idx] != col_flat_toy_data[idx]):
                col_squared_error.append((col_imputed_data[idx] - col_ground_truth[idx])**2)
                col_replacements.append(col_imputed_data[idx])
            
        norm_col_nrmse = np.sqrt(np.nanmean(col_squared_error)) / (max_col-min_col)
        total_col_imputed_data.append(np.concatenate(col_replacements))
        squared_error.append(norm_col_nrmse)
            
            
        
        
    
    nrmse = (np.nanmean(squared_error))
    print(nrmse)
    #roc =  roc_auc_score(flat_ground_truth[np.argwhere(np.isnan(flat_toy_data))[:,0]], 
    #                     flat_imputed_data[np.argwhere(np.isnan(flat_toy_data))[:,0]])
    
    
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(imputed_data_x, 
                                                            target, 
                                                            test_size=0.3, 
                                                            random_state=42)
    
    #instantiate the model
    log_regression = LogisticRegression()

    #fit the model using the training data
    log_regression.fit(X_train_2,y_train_2)
    
    
    y_pred_proba = log_regression.predict_proba(X_test_2)[::,1]
    fpr, tpr, _ = roc_curve(y_test_2,  y_pred_proba)
    auroc = roc_auc_score(y_test_2, y_pred_proba)
   
   
    
  
    
    
    return nrmse,fpr,tpr, auroc, total_col_imputed_data, squared_error

def evaluate_over_methods(methods, 
                          data, 
                          trials,
                          missing_rate,
                          missingness_type,
                          dataset_name,
                          hint_rate_vals, 
                          alpha_vals, 
                          ground_truth,
                          target,
                          current_features = None,
                          prev_columnwize_squared_errors = None,
                          prev_methods = None):
    
    '''
    evaluates multiple methods at a time on the same data and same ground truth over multiple trials (if specified)
    
    INPUTS
    
    methods : string array of methods
    data : data from original data source
    t : thresholding for missingness (i.e. higher t  = higher amount of missing value)
    trials : # of trials
    missingness_type : the type of missingness in all caps as a string.
    
    
    '''
    
    print(" ")
    print(missingness_type)
    nrmses = []
    fprs = []
    tprs = []
    aurocs = []
    trial_results = []
    columnwise_squared_errs = []
    
    n_samples = len(data[:,0])
    indices = range(n_samples)
    
   
    
    (X_train, X_test, y_train, y_test, indices_train, indices_test) = train_test_split(data, 
                                                            ground_truth, 
                                                            indices,
                                                            test_size=0.3, 
                                                            random_state=42)
    
    method_by_col = []
    idxs = []
    
    if(np.any(current_features) == True):
        scaler = StandardScaler()
        col_X_train = [X_train[:,i] for i in range(np.shape(X_train)[1])]
        
        col_X_train_edited = [np.reshape(scaler.fit_transform(np.reshape(x[~np.isnan(x)], (-1,1))), (1,-1)).squeeze() for x in col_X_train]
        current_features_edited = [np.reshape(scaler.fit_transform(np.reshape(x[~np.isnan(x)], (-1,1))), (1,-1)).squeeze() for x in current_features]
        
        for test_train in col_X_train_edited:
            #diff_from_norm_train = np.sum(list(kstest(test_train, 'norm')))
            #diff_from_norm_curr = [np.sum(list(kstest(x, 'norm'))) for x in current_features_edited]
            print(current_features_edited[0][0])
            print(test_train[0])
            print(np.shape(current_features_edited[0]))
            print(np.shape(test_train))
            err = [list(anderson_ksamp([test_train,x]))[0] for x in current_features_edited]
            minimum = np.min(err)
            
            idxs.append(np.where(err == minimum))
        print(idxs)
        for idx in idxs:
           
            print(np.shape((prev_columnwize_squared_errors)))
            
            method_squared_errs = [prev_columnwize_squared_errors[i][idx] for i in range(len(prev_methods))]
            print(idx)
            
            method_by_col.append(np.argmax(method_squared_errs))
        print(method_by_col)
    
        
            
                
    
    for method in methods:
        print(" ")
        print(method)
        nrmse_trial = []
        fpr_trial = []
        tpr_trial = []
        auroc_trial = []
        
        
        for j in range(trials):
            print(" ")
            print(f'trial # {j}')
            nrmse,fpr,tpr, auroc, trials_df, squared_errors = evaluate(X_train, 
                                                                       X_test, 
                                                                       y_train, 
                                                                       y_test, 
                                                                       missing_rate = missing_rate, 
                                                                       missingness = missingness_type,
                                                                       trial_num = j+1,
                                                                       dataset_name= dataset_name,
                                                                       method = method, 
                                                                       hint_rate_vals = hint_rate_vals,
                                                                       alpha_vals = alpha_vals, 
                                                                       target = target[indices_train],
                                                                       method_by_col = method_by_col)
            nrmse_trial.append(nrmse)
            fpr_trial.append(fpr)
            tpr_trial.append(tpr)
            auroc_trial.append(auroc)
            trial_results.append(trials_df)
            columnwise_squared_errs.append(squared_errors)
            #print(trials_df[0] - trials_df[1])
        nrmses.append([method, np.mean(nrmse_trial), np.std(nrmse_trial), trials])
        fprs.append([method, np.mean(fpr_trial,axis = 0), np.std(fpr_trial, axis = 0), trials])
        tprs.append([method, np.mean(tpr_trial,axis = 0), np.std(tpr_trial, axis = 0), trials])
        aurocs.append([method, np.mean(auroc_trial,axis = 0), np.std(auroc_trial, axis = 0), trials])
    num_columns = len(trial_results[0])
    
    
    for col in range(num_columns):
        fig,axes = plt.subplots(1,2)
        if(np.any(current_features) == True):
            fig,axes = plt.subplots(1,3)
        hist_data = [trial_results[i][col] for i in range(len(methods))]
        method_squared_errs = [columnwise_squared_errs[i][col] for i in range(len(methods))]
        axes[0].hist(y_train[:,col], alpha = 0.75)
        axes[0].set_title('true distribution (red = X_train)')
        axes[0].hist(X_train[:,col], alpha = 1, color = 'red')
        for idx in range(len(hist_data)):
            squared_err = np.round_(method_squared_errs[idx],3)
            
            if(methods[idx] == 'median' or methods[idx] == 'mean'):
                val = np.round_(hist_data[idx][0], 3)
                axes[1].hist(hist_data[idx], label = f'{methods[idx]} : {val} w/ err : {squared_err}', alpha = 0)
            elif(methods[idx] == 'combination'):
                combination_method = prev_methods[method_by_col[col]]
                axes[1].hist(hist_data[idx], label = f'{methods[idx]} : {combination_method} w/ err : {squared_err}', alpha = 0)
            else:
                axes[1].hist(hist_data[idx], label = f'{methods[idx]} w/ err : {squared_err}', alpha = 0.5)
        axes[1].legend()
        fig.suptitle(f'column {col} under missingness_type {missingness_type} ', fontsize=12)
        if(np.any(current_features) == True):
            matched_idx = idxs[col]
            axes[2].hist(y_train[:,col], alpha = 0.75)
            axes[2].hist(np.reshape(current_features[matched_idx], (-1,1)), alpha = 1, color = 'red')
            axes[2].set_title(f'matched distribution ({matched_idx})')
            
        
        
             
        path = r'C:\Users\potat\Downloads\mimic-lstm-master\mimic-lstm-master\toy_dataset_experiments'
    
        if(np.any(current_features) == True):
            
            path = r'C:\Users\potat\Downloads\mimic-lstm-master\mimic-lstm-master\toy_dataset_experiments\combination'
            
            if not(os.path.isdir(path)):
                os.mkdir(path)
            
            path = r'C:\Users\potat\Downloads\mimic-lstm-master\mimic-lstm-master\toy_dataset_experiments\combination/hist'
        
        else:
            
            if not(os.path.isdir(path)):
                os.mkdir(path)
            
            path = r'C:\Users\potat\Downloads\mimic-lstm-master\mimic-lstm-master\toy_dataset_experiments\hist'
    
        if not(os.path.isdir(path)):
            os.mkdir(path)
            
        path = os.path.join(path, dataset_name)
        if not(os.path.isdir(path)):
            os.mkdir(path)
    
        path = os.path.join(path, str(missingness_type))
        if not(os.path.isdir(path)):
            os.mkdir(path)
    
        path = os.path.join(path, str(missing_rate))
        if not(os.path.isdir(path)):
            os.mkdir(path)
        
    
        fig.savefig(os.path.join(path,f'col_{col}.jpg'))
    return nrmses, fprs,tprs, aurocs, X_train, columnwise_squared_errs
    