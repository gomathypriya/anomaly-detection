#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:59:58 2018

@author: Gomathypriya Dhanapal
@University: National University of Singapore
Code Description: Variable Standardisation and Derived Variables
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Read Input CSV dataset from working directory
# Arrange csv file columns in the below order
# First column is target (Y varaible), followed by numerical and last columns are catgorical
data = pd.read_csv("Rawdata.csv") # Input source file name

# One hot encoding technique is used to encode categorical integer features using a one-hot aka one-of-K scheme
def one_hot(data, tar_col, num_cols, cat_cols):
    categorical_columns = data.columns.values.tolist()[tar_col+num_cols:]
    numeric_columns = data.columns.values.tolist()[tar_col:tar_col+num_cols]
    for param in categorical_columns:    # Impute 'missing'text for categorical parameters
        idx = data[param].isnull().tolist()
        data.loc[idx,param] = 'missing'
        one_hot_data = pd.get_dummies(data,columns=categorical_columns)
        return one_hot_data, numeric_columns

# Standardization or z-scores normalization converts all indicators to a common scale with an average of zero and standard deviation of one. 
def zscore(x,mv,sd):
    return (x-mv)/sd


# one Hot encoded categorical columns: one_hot_data, numeric_columns = one_hot(data, tar_col=1, num_cols=14, cat_cols=6)
tar_col=1 # number of target variables
num_cols=14 # number of numeric columns
cat_cols=6 # number of categorical columns

categorical_columns = data.columns.values.tolist()[tar_col+num_cols:]
numeric_columns = data.columns.values.tolist()[tar_col:tar_col+num_cols]
# impute 'missing'text for categorical parameters
for param in categorical_columns:
    idx = data[param].isnull().tolist()
    data.loc[idx,param] = 'missing'
     
# one_hot_encoding
one_hot_data = pd.get_dummies(data,columns=categorical_columns)

# test train split
train, test = train_test_split(one_hot_data, test_size=0.3)

# Mean imputing
for param in numeric_columns:
    mean_value = train[param].mean()
    idx = train[param].isnull().tolist()
    train.loc[idx,param] = mean_value
        
    idx1 = test[param].isnull().tolist()
    test.loc[idx1,param] = mean_value

# z-score normalizations
for param in numeric_columns:
    mean_value = train[param].mean()
    sd_value = train[param].std()
    
    train[param] = train[param].apply(zscore,args={mean_value,sd_value}) #apply zscore to training data
    test[param] = test[param].apply(zscore,args={mean_value,sd_value}) #apply zscore to test data
    
########################################################################################
# Derived Varaibles
## transforming GV Energy and Air Bag Deployment
# converted to 4 categorical values
# 0 -- airbag not deployed + low GV energy
# 1 -- airbag not deployed + high GV energy
# 2 -- airbag deployed + low GV energy
# 3 -- airbag deployed + high GV energy  

def conv_binary(x,th):
    if x >= th:
        return 1
    else:
        return 0

mean_GV_Energy = train['GV_ENERGY'].mean()
train['Energy_bag'] = train["GV_ENERGY"].apply(conv_binary,th = mean_GV_Energy)
test['Energy_bag'] = test["GV_ENERGY"].apply(conv_binary,th = mean_GV_Energy)

train['Energy_bag'] = train['OA_BAGDEPLY_Deployed']+2*train['Energy_bag']
test['Energy_bag'] = test['OA_BAGDEPLY_Deployed']+2*test['Energy_bag']


########################################################################################
## Transforming Airbag Deployement and Seatbelt used into 4 categorical values
# converted to 4 categorical values
# 0 -- airbag not deployed + seatbelt not used
# 1 -- airbag not deployed + seatbelt used
# 2 -- airbag deployed + seatbelt not used
# 3 -- airbag deployed + seatbelt used

train['Airbag_Seatbelt'] = 2*train['OA_BAGDEPLY_Deployed']+train['OA_MANUSE_1.0']
test['Airbag_Seatbelt'] = 2*test['OA_BAGDEPLY_Deployed']+test['OA_MANUSE_1.0']


########################################################################################
# Transform VE_GAD1 from 4 categorical values to 2
# 1 -- VE_GAD1 is Front or Left
# 0 -- VE_GAD1 is Back or right

train['VE_GAD_2c'] = train['VE_GAD1_Front'] + train['VE_GAD1_Left'] 
# Possible values can be 0, 1, 2. Map 1 and 2 as high collision
test['VE_GAD_2c'] = test['VE_GAD1_Front'] + test['VE_GAD1_Left'] 

train['VE_GAD_2c'] = train["VE_GAD_2c"].apply(conv_binary,th = 1)
test['VE_GAD_2c'] = test["VE_GAD_2c"].apply(conv_binary,th = 1)


########################################################################################
# Transform PDOF into 2 categorical variables, angle closer to driver or not
# train['VE_PDOF_TR']
# 1 -- PDOF_TR is between 45 to 135 degree (driver side)
# 0 -- PDOF_TR outside the interval [45, 135]

def driver_angle(x):
    if (x>=45) & (x<=135):
        return 1
    else:
        return 0
    
train['Driver_side'] = train['VE_PDOF_TR'].apply(driver_angle)
test['Driver_side'] = test['VE_PDOF_TR'].apply(driver_angle)



########################################################################################
# Transform age into 3 categories based young, medium, and old
# transform age into 3 categories
# 0 -- Age is between min to min + range/3
# 1 -- Age is between min+range/3 to max-range/3
# 2 -- Age is greater than max-range/3

min_age = train['OA_AGE'].min()
max_age = train['OA_AGE'].max()


def cov_ter(x,min_v, max_v):
    age_range = max_v - min_v
    th = age_range/3.0
    if x<=min_v+th:
        return 0
    elif x > max_v-th:
        return 2
    else:
        return 1

train['Age_3c'] = train['OA_AGE'].apply(cov_ter, min_v = min_age, max_v = max_age)
test['Age_3c'] = test['OA_AGE'].apply(cov_ter, min_v = min_age, max_v = max_age)

########################################################################################
swap_columns = ['OA_MAIS', 'GV_CURBWGT', 'GV_DVLAT', 'GV_DVLONG','GV_LANES', 'GV_OTVEHWGT', 'GV_SPLIMIT', 
 'GV_MODELYR_2000', 'GV_MODELYR_2001','GV_MODELYR_2002', 'GV_MODELYR_2003', 'GV_MODELYR_2004',
 'GV_MODELYR_2005', 'GV_MODELYR_2006', 'GV_MODELYR_2007','GV_MODELYR_2008', 'GV_MODELYR_2009', 
 'GV_MODELYR_2010','GV_MODELYR_2011', 'GV_MODELYR_2012', 
 'GV_WGTCDTR_Passenger Car','GV_WGTCDTR_Truck (<=10000 lbs.)', 'GV_WGTCDTR_Truck (<=6000 lbs.)',
 'Energy_bag','Airbag_Seatbelt', 'VE_GAD_2c', 'Driver_side', 'Age_3c',
 'VE_PDOF_TR','GV_ENERGY','OA_HEIGHT',
 'OA_SEX_Female','OA_SEX_Male', 'OA_SEX_missing','OA_AGE', 'OA_WEIGHT',
 'GV_FOOTPRINT','VE_ORIGAVTW', 'VE_WHEELBAS','OA_BAGDEPLY_Deployed', 'OA_BAGDEPLY_Not Deployed', 
 'OA_MANUSE_0.0','OA_MANUSE_1.0', 'OA_MANUSE_missing',
 'VE_GAD1_Front', 'VE_GAD1_Left','VE_GAD1_Rear','VE_GAD1_Right', 'VE_GAD1_missing']

tr1 = train[swap_columns]
te1 = test[swap_columns]

# Create csv files train  dataset and test dataset with derived columns
tr1.to_csv('train_derived.csv',index = False)
te1.to_csv('test_derived.csv', index = False)
