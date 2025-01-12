#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

SEED = 1


### Import data

raw_data_df = pd.read_csv('loan_data_2007_2014.csv', index_col=0)
raw_data_df
raw_data_df.info()
raw_data_df.shape


### Data cleaning & EDA

# Remove duplicate rows
# Check if there are duplicated rows
raw_data_df['member_id'].duplicated().any()


# Drop useless columns
useless_cols = ['id', 
                'member_id', 
                'url',
                'desc',
                'emp_title',
                'pymnt_plan',
                'title',
                'funded_amnt_inv',
                'addr_state',
                'zip_code',
                'pymnt_plan',
                'last_pymnt_d',
                'next_pymnt_d']
clean_df = raw_data_df.drop(columns=useless_cols)


## String manipulation

#list(clean_df['term'].unique())

# term
list(clean_df['term'].unique())
# In column 'term', remove string 'months'
clean_df['term'] = clean_df['term'].str.replace('months', '')
clean_df['term'] = clean_df['term'].astype(int)
list(clean_df['term'].unique())

# emp_length
list(clean_df['emp_length'].unique())
# In column 'emp_length', remove text
clean_df['emp_length'] = clean_df['emp_length'].str.replace('+', '')
clean_df['emp_length'] = clean_df['emp_length'].str.replace(' years', '')
clean_df['emp_length'] = clean_df['emp_length'].str.replace('< 1 year', '0')
clean_df['emp_length'] = clean_df['emp_length'].str.replace(' year', '')
list(clean_df['emp_length'].unique())


## Check null values

# pd.set_option('display.max_rows', 500)

(clean_df.isnull().sum()/len(clean_df))*100


# Drop columns with null values above 70%

null_cols = ['annual_inc_joint',
            'dti_joint',
            'verification_status_joint',
            'open_acc_6m',
            'open_il_6m',
            'open_il_12m',
            'open_il_24m',
            'mths_since_rcnt_il',
            'mths_since_last_record',
            'mths_since_last_major_derog',
            'total_bal_il',
            'il_util',
            'open_rv_12m',
            'open_rv_24m',
            'max_bal_bc',
            'all_util',
            'inq_fi',
            'total_cu_tl',
            'inq_last_12m']

clean_df = clean_df.drop(columns=null_cols)


# Impute null values with median
for col in ['annual_inc', 
            'revol_util',
            'tot_cur_bal',
            'total_rev_hi_lim']:
    clean_df[col].fillna(clean_df[col].median(), inplace=True)
    clean_df[col] = clean_df[col].astype(int)

# Impute null values with mode
for col in ['emp_length', 
            'earliest_cr_line',
            'last_credit_pull_d']:
    clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)

# Impute null values as 0
for col in ['delinq_2yrs',
            'inq_last_6mths',
            'open_acc',
            'pub_rec',
            'total_acc',
            'open_acc',
            'pub_rec',
            'collections_12_mths_ex_med',
            'acc_now_delinq',
            'tot_coll_amt',
            'mths_since_last_delinq']:
    clean_df[col].fillna(0, inplace=True)
    clean_df[col] = clean_df[col].astype(int)

# Final check for null values
clean_df.isnull().any()


### Set target variable

# Loan status categories
list(clean_df['loan_status'].unique())

# high_risk = ['Does not meet the credit policy. Status:Charged Off',
#              'Charged Off',
#              'Default',
#              'Late (31-120 days)',
#              'In Grace Period',
#              'Late (16-30 days)']
low_risk = ['Does not meet the credit policy. Status:Fully Paid',
            'Fully Paid', 
            'Current']

# Change loan status to its corresponding labels
clean_df['loan_status'] = clean_df['loan_status'].apply(lambda s: 0 if s in low_risk else 1)

list(clean_df['loan_status'].unique())


## Check data correlation

# Create correlation matrix
corr_matrix = clean_df.corr()

# Display correlation matrix
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix)
plt.show()

# Remove low correlating data
clean_df = clean_df.drop(columns=['policy_code','recoveries','collection_recovery_fee'])


### Feature engineering

# Separate features with target variable
y = clean_df['loan_status']
X = clean_df.drop(columns=['loan_status'])


## Encode categorical data

# Get columns with categorical data
categorical_cols = list(X.select_dtypes(include=['object']).columns)
categorical_cols

# Encode categorical data
X = pd.get_dummies(X, columns=categorical_cols)

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Encode target variable
lab_encoder = LabelEncoder()
y = lab_encoder.fit_transform(y)

# Split data to train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('\nX_test:', X_test.shape)
print('y_test:', y_test.shape)


### Build model (Random Forest Classifier)

# Build model
rfc_model = RandomForestClassifier(random_state=SEED, n_jobs=-1, verbose=1)

# Train model
rfc_model.fit(X_train, y_train)

### Model evaluation

def print_metrics(y_test, y_preds):
    metrics = {'accuracy': accuracy_score(y_test, y_preds),
               'precision':  precision_score(y_test, y_preds, average='macro'),
               'recall': recall_score(y_test, y_preds, average='macro'),
               'roc_auc': roc_auc_score(y_test, y_preds, average='macro')}

    print("Results on testing set")
    print("----------------------")
    print("Accuracy score:", metrics['accuracy'])
    print("Precision score:", metrics['precision'])
    print("Recall score:", metrics['recall'])
    print("ROC-AUC score:", metrics['roc_auc'])

print_metrics(y_test, rfc_model.predict(X_test))

