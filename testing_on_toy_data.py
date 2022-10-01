# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 23:12:42 2022

@author: potat
"""

import pickle
import math
import re
import csv
import concurrent.futures 
import os
from functools import reduce

from gain import gain
from operator import add
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from uci_helper import generate_mcar_mar_mnar, convert_to_mcar, convert_to_mar, convert_to_mnar,generate_uniform_matrix,convert_to_numerical, translate_to_dataset, convert_to_mar_mnar
from testing_utils import evaluate_over_methods
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kstest

print(os.getcwd() + "\data\toy_datasets\mnar_toy_data.csv")

mnar_toy_dataset = pd.read_csv(r"data\toy_datasets\mnar_toy_data.csv")
mar_toy_dataset = pd.read_csv(r"data\toy_datasets\mar_toy_data.csv")
mcar_toy_dataset = pd.read_csv(r"data\toy_datasets\mcar_toy_data.csv")

spam = pd.read_csv(r"data\uci_machine_learning_data\spam.csv")
#drop_target(spam,'target')

breast = pd.read_csv(r"data\uci_machine_learning_data\breast_cancer_uci.csv")
#drop_target(breast, 'Class')
breast = convert_to_numerical(breast, breast.columns)

messidor = pd.read_csv(r"data\uci_machine_learning_data\messidor_features.csv")
#drop_target(messidor,'Class')

letter = pd.read_csv(r"data\uci_machine_learning_data\letter.csv")

credit_card = pd.read_csv(r"data\uci_machine_learning_data\UCI_Credit_Card.csv")
credit_card = credit_card.drop(['ID'],axis = 1)
#drop_target(credit_card,'default.payment.next.month')


#mcar_mar_ground_truth = pd.read_csv(r"C:\Users\potat\Downloads\mimic-lstm-master\mimic-lstm-master\data\toy_datasets\ground_truth_mar_mcar.csv")
ground_truth = pd.read_csv(r"data\toy_datasets\ground_truth.csv")

gain_parameters = {'batch_size': 100,
                     'hint_rate': 0.1,
                     'alpha': 200,
                     'iterations': 10000,
                     'mode' : 'Normal'}
    
def exact_missingness(missingness_copy, intercept_pct, missingness, ground_truth, split_1, split_2, z):
    while((missingness_copy- intercept_pct) < missingness):
        missingness_copy = missingness_copy + .01
        data,intercept_pct = convert_to_mar_mnar(ground_truth,missingness*split_1,missingness*split_2,z)
    return data,intercept_pct, missingness_copy
                  

def evaluate_over_missingness(methods, 
                          data_no_missing, 
                          trials,  
                          hint_rate_vals, 
                          alpha_vals, 
                          missingnesses,
                          missing_types,
                          dataset_name,
                          target,
                          prev_features = None,
                          prev_lookup_table = None,
                          prev_methods = None,
                          prev_columnwize_errs = None,
                          eval_over = None
                          ):
  '''evaluates methods over different missing types specified and different missing rates where
  there is a hyperparameter search in GAIN and MIM-GAIN over hint_rate_vals and alpha_vals'''
  lookup_table = []
  nrmses_over_vals = []
  tprs_over_vals = []
  fprs_over_vals = []
  aurocs_over_vals = []
  columnwize_squared_errors_over_vals = []
  intercept_pcts_over_vals = []
  data_over_vals = []
  
  scaler = MinMaxScaler()
  
  data_no_missing[data_no_missing.columns] = scaler.fit_transform(data_no_missing[data_no_missing.columns])
  data_no_missing.to_csv('scaled_dataset.csv')
  
  
      
  
  ground_truth = data_no_missing
  z = generate_uniform_matrix(ground_truth)
  for missingness in missingnesses:
      print(missingness)
      for missing_type in missing_types:
          data = []
          if(missing_type == 'MNAR'):
              data = convert_to_mnar(ground_truth, missingness, z)
              data = translate_to_dataset(data,ground_truth)
              intercept_pcts_over_vals.append("")
          elif(missing_type == 'MAR'):
              data = convert_to_mar(ground_truth,missingness,z)
              data = translate_to_dataset(data,ground_truth)
              intercept_pcts_over_vals.append("")
          elif(missing_type == 'MCAR'):
              data = convert_to_mcar(ground_truth,missingness,z)
              data = translate_to_dataset(data,ground_truth)
              intercept_pcts_over_vals.append("")
          elif(missing_type.startswith('EHR')):
              #half and half
              split_1 = float(missing_type[4:7]) #for mar
              split_2 = float(missing_type[9:])  #for mnar
              print(f'mar split: {split_1}, mnar split: {split_2}')
              data,intercept_pct = convert_to_mar_mnar(ground_truth,missingness*split_1,missingness*split_2,z)
              missingness_copy = missingness
              data,intercept_pct, missingness_copy = exact_missingness(missingness_copy, intercept_pct, missingness, ground_truth, split_1, split_2, z)
              true_missing = (missingness_copy - intercept_pct)
              intercept_pcts_over_vals.append([true_missing, intercept_pct])
              print(f"percent of missing is {true_missing} which is {missingness_copy} - {intercept_pct}")
              data = translate_to_dataset(data,ground_truth)
          
          current_features = []
          current_columnwise_errs = []
          if(np.any(prev_features) == True and not(prev_lookup_table.empty) == True):
              split_missing_type = str(missing_type).split()[0]
              print(split_missing_type.split())
              print(prev_lookup_table)
              
              qry = ''
              qry = qry + 'missingness == ' + str(missingness)
              qry = qry + ' and '
              qry = qry + 'missing_type == "' + str(missing_type) +'" '
              
              idx = int(list(prev_lookup_table.query(qry).index)[0])
              print(idx)
              current_features = prev_features[idx]
              current_columnwise_errs = prev_columnwize_errs[idx]
              print(np.shape(current_features))
              '''for j in current_features:
                  print(j)
                  ks = (kstest(j[~np.isnan(j)],'norm', N = np.shape(current_features)[0]))
                  print(ks)
              break
              ks_test = [list(kstest(x,'norm'))[1] for x in current_features]
              print(ks_test)
              break'''
              
              
          nrmses, fprs,tprs, aurocs, X_train, columnwise_squared_errs= evaluate_over_methods(methods, 
                                                                                             data, 
                                                                                             trials, 
                                                                                             missingness, 
                                                                                             missing_type, 
                                                                                             dataset_name,
                                                                                             hint_rate_vals, 
                                                                                             alpha_vals, 
                                                                                             ground_truth.values, 
                                                                                             target, 
                                                                                             current_features,
                                                                                             current_columnwise_errs , 
                                                                                             prev_methods)
          
          nrmses_over_vals.append(nrmses)
          fprs_over_vals.append(fprs)
          tprs_over_vals.append(tprs)
          aurocs_over_vals.append(aurocs)
          num_features = np.shape(X_train)[1]
          data_over_vals.append(np.array([X_train[:,feature] for feature in range(num_features)]))
          columnwize_squared_errors_over_vals.append(np.array(columnwise_squared_errs))
          #trial_results_over_vals.append(trial_results)
          lookup_table.append([missingness, missing_type])
      
  return lookup_table, nrmses_over_vals, fprs_over_vals, tprs_over_vals, aurocs_over_vals, intercept_pcts_over_vals, data_over_vals, columnwize_squared_errors_over_vals
    
hint_rate_vals   = [1]    
alpha_vals = [0,50,100]
target_col_credit = 'default.payment.next.month' #credit card
target_col_breast = 'Class'
target_col_spam = 'target'
prev_methods = ["GAIN","median"]

target_col_mess = 'Class' #messidor

lookup_table, nrmses_over_vals, fprs_over_vals, tprs_over_vals, aurocs_over_vals, intercept_pcts_over_vals, data_over_vals, columnwize_squared_errors_over_vals = evaluate_over_missingness(["GAIN","median"], 
                                                                                           credit_card.drop(target_col_credit,axis = 1), 1, 
                          hint_rate_vals, 
                          alpha_vals, 
                          [0.8],
                          ['EHR_0.2_0.8', 'EHR_0.5_0.5', 'EHR_0.8_0.2'],
                          'credit_card',
                          credit_card[target_col_credit].values
                          
                          )


lookup_table = np.array(lookup_table)
prev_lookup_table = pd.DataFrame({'missingness' : lookup_table[:,0], 'missing_type' : lookup_table[:,1]})
prev_lookup_table['missingness'] = prev_lookup_table['missingness'].astype(float)

extra_lookup_table, extra_nrmses_over_vals, extra_fprs_over_vals, extra_tprs_over_vals, extra_aurocs_over_vals, extra_intercept_pcts_over_vals, extra_data_over_vals, extra_columnwize_squared_errors_over_vals = evaluate_over_missingness(["combination","GAIN", "median"], 
                                                                                           spam.drop(target_col_spam,axis = 1), 1, 
                          hint_rate_vals, 
                          alpha_vals, 
                          [0.8],
                          ['EHR_0.2_0.8', 'EHR_0.5_0.5', 'EHR_0.8_0.2'],
                          'spam',
                          spam[target_col_spam].values,
                          prev_features = data_over_vals,
                          prev_lookup_table = prev_lookup_table,
                          prev_methods = prev_methods,
                          prev_columnwize_errs= columnwize_squared_errors_over_vals
                          )


#nrmses_mnar, aucs_mnar, val = evaluate_over_methods(["MIM-GAIN","GAIN","mean","median"],data,2, 'MNAR', hint_rate_vals, alpha_vals, ground_truth = ground_truth.iloc[:,1:].values)
#nrmses_mar, aucs_mar, val = evaluate_over_methods(["MIM-GAIN","GAIN","mean","median"], mar_toy_dataset.iloc[:,1:],2, 'MAR', hint_rate_vals, alpha_vals)
#nrmses_mcar, aucs_mcar, val = evaluate_over_methods(["MIM-GAIN","GAIN","mean","median"], mcar_toy_dataset.iloc[:,1:],2, 'MCAR', hint_rate_vals, alpha_vals)
        
#sensitive to hyperparameter tuning


