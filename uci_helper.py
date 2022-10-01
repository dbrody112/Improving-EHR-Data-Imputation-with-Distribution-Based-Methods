# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:09:58 2022

@author: potat
"""

import pandas as pd
import numpy as np



def generate_uniform_matrix(dataset):
    cols = len(dataset.columns)
    rows = len(dataset)
    
    z = np.random.uniform(size = (rows,cols))
    
    return z

def convert_to_mcar(dataset,t, z):
    binary_z = (z <= t).astype('float')
    
    return binary_z
    

def convert_to_mar(dataset,t, z):
    binary_z = (z <= t).astype('int')
    size = 11
    if(len(dataset.columns) < size):
        size = len(dataset.columns)
    x = np.random.randint(0,len(dataset.columns)-1,size)
    feature_col = dataset.columns[0]
    median_val = np.median(dataset[feature_col].values)
    median_z = (dataset[feature_col].values <= median_val).astype('int')
    for idx in x[1:size-1]:
        binary_z[:,idx] = binary_z[:,idx] & median_z
    binary_z = binary_z.astype('float')
    
    return binary_z
    

def convert_to_mnar(dataset,t,z):
    binary_z = (z <= t).astype('int')
    size = 5
    if(len(dataset.columns) < size):
        size = len(dataset.columns)
    x = np.random.randint(0, len(dataset.columns)-1, size)
    for idx in x:
        median_val = np.median(dataset[dataset.columns[idx]].values)
        median_z = (dataset[dataset.columns[idx]].values <= median_val).astype('int')
        binary_z[:,idx] = binary_z[:,idx] & median_z
    binary_z = binary_z.astype('float')
    
    return binary_z
    

def convert_to_mar_mnar(dataset,t1,t2,z):
    #t1 + t2 must be less than 1
    #t1 percent of mar
    #t2 percent of mnar
    #there will inevitably be some intersection so that will be taken into account
    mar_dataset = convert_to_mar(dataset, t1, z)
    
    
    mnar_dataset = convert_to_mar(dataset,t2,z)
    
    
    cols = len(dataset.columns)
    rows = len(dataset)
    total_ones = (np.sum(mar_dataset)+np.sum(mnar_dataset))
    
    combination_binary_z = mar_dataset.astype('int') & mnar_dataset.astype('int')
    
    intercepted_points_percent = abs(np.sum(combination_binary_z) - total_ones) / (cols*rows)
    
    return combination_binary_z.astype('float'), intercepted_points_percent
    
def translate_to_dataset(binary_z,dataset):
    print(binary_z)
    binary_z[binary_z == 1] = np.nan
    binary_z[binary_z == 0] = 1
    return binary_z *  dataset.values
    


def convert_to_numerical(dataset, categorical_cols):
    for col in categorical_cols:
        dataset[col] = dataset[col].astype('category')
        dataset[col] = dataset[col].cat.codes
    return dataset

def generate_mcar_mar_mnar(dataset,t):
    z = generate_uniform_matrix(dataset)
    
    mcar_dataset = convert_to_mcar(dataset,t,z)
    mcar_dataset = translate_to_dataset(mcar_dataset,dataset)
    
    mar_dataset = convert_to_mcar(dataset,t,z)
    mar_dataset = translate_to_dataset(mar_dataset,dataset)
    
    mnar_dataset = convert_to_mcar(dataset,t,z)
    mnar_dataset = translate_to_dataset(mnar_dataset,dataset)
    
    
    return mcar_dataset, mnar_dataset, mar_dataset


#4:7
#9:
print(float("EHR_0.2_0.8"[9:]))

    
    
    
    