'''
Title: Student Hold Predictor
'''

#%% Libraries
import os
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
import tkinter as tk
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


#%%% Data Import and Combination
os.chdir('C:\\Users\\bradm\\OneDrive\\Projects\\Hold_Predictor')

rest_df = pd.read_csv('Active Students Restrictions.csv') # Restrictions
df = pd.read_csv('Active Students By Major.xlsx') # main

#%% data cleaning
df.rename(columns = {'PersonID':'StudentID'}, inplace = True) # Rename 'PersonID' to 'StudentID' in df
rest_df.drop(["StartTerm1", "FirstName", "LastName", "AcadProgramSchool","AcadProgramDepartment", "AcadProgramDescription", "AcadProgramStatusCode", "AcadProgramStatusDescription"], axis = 1, inplace=True)
df_2 = pd.merge(rest_df, df, on='StudentID', how='right') #join
df_2.loc[(df_2.Restriction == 'FSFIN'), 'Restriction'] = 1 # make Restriction 1
df_2['Restriction'] = df_2['Restriction'].fillna(0) #Replace nulls with 0

#%% Dropping useless variables
df_2 = df_2.drop(['StudentID', 'FirstName', 'MiddleName', 'LastName', 'BirthDate',
'Age', 'AlienStatusCode', 'AlienStatusDescription', 'ImmigrationStatusCode', 'ImmigrationStatusDescription', 'PrimaryCitizenshipCode',
'PrimaryCitizenshipDescription', 'DenominationDescription', 'Sport', 'MajorDepartment', 'MajorCode', 'MajorDescription', 'AcadProgramCode',
'AcadProgramDescription', 'AcadProgramStatusDescription', 'AnticipatedCompletionDate', 'EnrollStatusCode', 'EnrollStatusDescription', 'CatalogCode', 
'StartTerm', 'LastTermRegistered', 'AddressLine1', 'AddressLine2', 'City', 'ZipCode', 'CountryCode', 'CountryName', 'BusinessPhone', 'HomePhone',
'MobilePhone', 'PersonalEmail', 'CampusEmail', 'AdvisorName', 'ProgramAdvised', 'RestrictionDescription','HoldStartDate', 'Textbox14', 'Textbox30',
'Textbox38', 'Textbox42', 'PrivacyCode', 'PrivacyCode', 'PrivacyCodeDescription', 'AcadLevelCode', 'EthnicGroupDescription', 'TermRegCreds',
'OneTermPriorRegCreds', 'TwoTermsPriorRegCreds', 'ThreeTermsPriorRegCreds'], axis = 1)

#%% Categorize variables
df_2['Gender'] = df_2.Gender.astype('category')
df_2['DenominationCode'] = df_2.DenominationCode.astype('category')
df_2['ClassCode'] = df_2.ClassCode.astype('category')
df_2['MajorSchool'] = df_2.MajorSchool.astype('category')
df_2['AcadProgramStatusCode'] = df_2.AcadProgramStatusCode.astype('category')
df_2['AdmitStatusDescription'] = df_2.AdmitStatusDescription.astype('category')
df_2['StateCode'] = df_2.StateCode.astype('category')
df_2['UGFreshmanTransfer'] = df_2.UGFreshmanTransfer.astype('category')

#%% Create Dummy Variables
df_2 = pd.get_dummies(df_2, drop_first=True)

#%% Fill remaining NaN with zero
df_2 = df_2.fillna(0)

#%% Model prep
X = df_2.drop(['Restriction'], axis = 1)
y = df_2.Restriction

#%% Normalize the Data
scaler = StandardScaler()
Xn = scaler.fit_transform(X)

#%% Training/Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =.3,
                                                    random_state=1234, stratify=y)


#%% Decision Tree 

# Create a model (object) for classification
dtm = DecisionTreeClassifier(random_state=5678)

# Build a decision tree classification model
dtm.fit(X_train, y_train)

# Make predictions using the test data
y_pred_dtm = dtm.predict(X_test)

# Build a confusion matrix and show the Classification Report
cm_dtm = metrics.confusion_matrix(y_test,y_pred_dtm, labels=[1,0])
print('\nConfusion Matrix for Decision Tree\n',cm_dtm)
print('\nClassification Report for Decision Tree\n')
print(metrics.classification_report(y_test,y_pred_dtm))


#%% Random Forest
rfcm = RandomForestClassifier(random_state=5678)
rfcm.fit(X_train, y_train)
y_pred_rfcm = rfcm.predict(X_test)

# %% Classification Report
cm_rfcm = metrics.confusion_matrix(y_test,y_pred_rfcm, labels=[1,0])
print('\nConfusion Matrix for Random Forest\n',cm_rfcm)
print('\nClassification Report for Random Forest\n')
print(metrics.classification_report(y_test,y_pred_rfcm))

#%% Gradient Boost
gbmc = GradientBoostingClassifier()
gbmc.fit(X_train, y_train)
y_pred_gbmc = gbmc.predict(X_test)

# %% Classification Report
cm_gbmc = metrics.confusion_matrix(y_test,y_pred_gbmc, labels=[1,0])
print('\nConfusion Matrix for Gradient Boost\n',cm_gbmc)
print('\nClassification Report for Gradient Boost\n')
print(metrics.classification_report(y_test,y_pred_gbmc))

#%% Neural Network Classifier
nnm = MLPClassifier(activation='relu', solver='adam', max_iter=200)
nnm.fit(X_train, y_train)
y_pred_nnm = nnm.predict(X_test)

# %% Classification Report
cm_nnm = metrics.confusion_matrix(y_test,y_pred_nnm, labels=[1,0])
print('\nConfusion Matrix for Neural Network\n',cm_nnm)
print('\nClassification Report for Neural Network\n')
print(metrics.classification_report(y_test,y_pred_nnm))

#%%

#%% Decision Tree Selection of parameter values
tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
CV_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=tree_para, cv= 5, scoring='recall')
CV_dt.fit(X_train, y_train)

#%% Best Params
CV_dt.best_params_


#%% Decision Tree with Best Parameters
# Create a model (object) for classification
dtm2 = DecisionTreeClassifier(random_state=5678, criterion='gini', max_depth=30)

# Build a decision tree classification model
dtm2.fit(X_train, y_train)

# Make predictions using the test data
y_pred_dtm2 = dtm2.predict(X_test)

# Build a confusion matrix and show the Classification Report
cm_dtm2 = metrics.confusion_matrix(y_test,y_pred_dtm2, labels=[1,0])
print('\nConfusion Matrix for Decision Tree\n',cm_dtm2)
print('\nClassification Report for Decision Tree\n')
print(metrics.classification_report(y_test,y_pred_dtm2))

# %%
