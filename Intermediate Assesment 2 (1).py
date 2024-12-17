#!/usr/bin/env python
# coding: utf-8

# # Problem Statement 

# Optimize the Loan eligibility process
# Predict Loan Eligibility for Dream Housing Finance company
# Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
# 
# Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers. 

# ## Data Dictionary

# ### Train file: CSVcontaining the customers for whom loan eligibility is known as 'Loan_Status'
# 
# - Variable	Description
# - Loan_ID	Unique Loan ID
# - Gender	Male/ Female
# - Married	Applicant married (Y/N)
# - Dependents	Number of dependents
# - Education	Applicant Education (Graduate/ Under Graduate)
# - Self_Employed	Self employed (Y/N)
# - ApplicantIncome	Applicant income
# - CoapplicantIncome	Coapplicant income
# - LoanAmount	Loan amount in thousands
# - Loan_Amount_Term	Term of loan in months
# - Credit_History	credit history meets guidelines
# - Property_Area	Urban/ Semi Urban/ Rural
# - Loan_Status	(Target) Loan approved (Y/N)
# 
# 
# ### Submission file format
# 
#  
# 
# - Variable	Description
# - Loan_ID	Unique Loan ID
# - Loan_Status	(Target) Loan approved (Y/N)
# 
# 

# ### Evaluation Metric
# Your model performance will be evaluated on the basis of your prediction of loan status for the test data (test.csv), which contains similar data-points as train except for the loan status to be predicted. Your submission needs to be in the format as shown in sample submission.
# 
# We at our end, have the actual loan status for the test dataset, against which your predictions will be evaluated. We will use the Accuracy value to judge your response.
# 
# 
# ### Public and Private Split
# Test file is further divided into Public (25%) and Private (75%)
# 
# - Your initial responses will be checked and scored on the Public data.
# - The final rankings would be based on your private score which will be published once the competition is over.
# 
# ### Guidelines for Final Submission
# Please ensure that your final submission includes the following:
# 
#  
# Solution file containing the predicted sales value in the test dataset (format is given in sample submission csv)
# Code file for reproducing the submission, note that it is mandatory to submit your code for a valid final submission

# # Importing Libraries

# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# # Loading Dataset

# In[75]:


# Importing training and testing datasets

df_train = pd.read_csv("train_ctrUa4K.csv")
df_test = pd.read_csv("test_lAUu6dG.csv")


# In[77]:


df_train.head()


# # Inspecting Dataset

# In[80]:


df_train.shape


# In[82]:


df_test.shape


# In[84]:


df_train.columns


# In[86]:


df_test.columns


# In[88]:


df_train.describe()


# In[90]:


df_train.info()


# # Checking for Duplicates

# In[93]:


df_train.duplicated().sum()


# In[95]:


# Zero duplicates found in rows training dataset


# In[97]:


df_test.duplicated().sum()


# In[99]:


# Zero duplicates found in rows in testing dataset


# # Checking Missing Values

# In[102]:


df_train.isna().sum()


# In[104]:


df_test.isna().sum()


# # Handling Missing Values 

# In[107]:


# Splitting into numerical and categorical columns

num_df = df_train.select_dtypes(include=['number'])
cat_df = df_train.select_dtypes(include = ['object'])

num_df_test = df_test.select_dtypes(include=['number'])
cat_df_test = df_test.select_dtypes(include = ['object'])


# In[109]:


num_columns = num_df.columns.tolist()
cat_columns = cat_df.columns.tolist()
print("Numerical columns:",num_columns)
print("Categorical columns:",cat_columns)

num_columns_test = num_df_test.columns.tolist()
cat_columns_test = cat_df_test.columns.tolist()
print("Numerical columns of test data:",num_columns_test)
print("Categorical columns of test data:",cat_columns_test)


# In[111]:


# Checking Correlation of Numerical Columns


# ## Checking Correlation of Numerical Columns

# In[114]:


sns.heatmap(num_df.corr(),annot= True)


# In[116]:


# Handling Missing Values in numerical columns


# In[118]:


# Checking Histogram for data imputation


# In[120]:


for col in num_columns:
    plt.hist(num_df[col])
    plt.xlabel(col)
    plt.ylabel('count')
    plt.title('Histogram of {}'.format(col))
    plt.show()


# In[121]:


# The data is right skewed. So lets impute the data using median values for numerical columns in both test and train datasets.


# In[122]:


# Filling missing values in numerical columns of train data
for col in num_columns:
  num_df[col] = num_df[col].fillna(num_df[col].median())

# Filling missing values in numerical columns of test data
for col in num_columns_test:
  num_df_test[col] = num_df_test[col].fillna(num_df_test[col].median())    


# In[126]:


# Checking missing values again in num_df
num_df.isna().sum()


# In[128]:


# Checking missing values again in num_df_test
num_df_test.isna().sum()


# In[130]:


# Handling Missing Values in categorical columns in train data
for col in cat_columns:
    cat_df[col] = cat_df[col].fillna(cat_df[col].mode()[0])

# Handling Missing Values in categorical columns in test data
for col in cat_columns_test:
    cat_df_test[col] = cat_df_test[col].fillna(cat_df_test[col].mode()[0])


# In[132]:


# Checking missing values again in cat_df
cat_df.isna().sum()


# In[134]:


# Checking missing values again in cat_df_test
cat_df_test.isna().sum()


# In[136]:


# Recombine both numerical and categorical columns
df_train = pd.concat([cat_df,num_df],axis=1)
df_test = pd.concat([cat_df_test,num_df_test],axis=1)


# In[138]:


df_train.head()


# In[140]:


df_test.head()


# In[142]:


# Removing the "+" string from Dependents column to make it into a numerical column
df_train['Dependents'] = df_train['Dependents'].str.replace('+', '')


# In[144]:


# Removing the "+" string from Dependents column to make it into a numerical column
df_test['Dependents'] = df_test['Dependents'].str.replace('+', '')


# In[146]:


df_train['Dependents'] = df_train['Dependents'].astype(float)
df_test['Dependents'] = df_test['Dependents'].astype(float)


# # Outlier Handling

# In[149]:


num_df.boxplot()
plt.xticks(rotation=45)
plt.show()


# In[151]:


num_df_test.boxplot()
plt.xticks(rotation=45)
plt.show()


# In[153]:


def remove_outliers(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    df[column_name] = df[column_name].clip(upper=upper_bound)
    df[column_name] = df[column_name].clip(lower=lower_bound)
    return df[column_name]


# In[155]:


for col in num_columns:
  num_df[col] = remove_outliers(num_df, col)


# In[157]:


for col in num_columns_test:
  num_df_test[col] = remove_outliers(num_df_test, col)


# In[159]:


num_df_test.boxplot()
plt.xticks(rotation=45)
plt.show()


# In[160]:


num_df.boxplot()
plt.xticks(rotation=45)
plt.show()


# # Checking Data Imbalance 

# In[164]:


# Plot class imbalance
sns.countplot(x='Loan_Status', data=df_train)
plt.title('Class Imbalance: Loan_status Distribution')
plt.xlabel('Loan_status')
plt.ylabel('Loan_status_count')
plt.xticks(rotation=45, ha='right')
plt.show()


# # Feature Encoding

# In[167]:


# One hot encoding for columns in training dataset
df_train = pd.get_dummies(df_train,columns=['Gender','Married','Education','Self_Employed','Property_Area'],dtype= int,drop_first=True)
df_train


# In[169]:


# One hot encoding for columns in testing dataset
df_test = pd.get_dummies(df_test,columns=['Gender','Married','Education','Self_Employed','Property_Area'],dtype= int,drop_first=True)
df_test


# In[171]:


Loan_ID =  pd.DataFrame(df_test['Loan_ID'])
Loan_ID


# In[173]:


df_train = df_train.drop('Loan_ID',axis = 1)

df_test = df_test.drop('Loan_ID',axis = 1)


# # Feature Scaling

# In[176]:


# min max scaling for faetures having non-gaussian distribution in training dataset
min_scaler = MinMaxScaler()
numerical_colms1 = ['Dependents','Loan_Amount_Term']
df_train[numerical_colms1] = min_scaler.fit_transform(df_train[numerical_colms1])
df_train


# In[178]:


# min max scaling for features having non gaussian distribution in testing dataset
min_scaler = MinMaxScaler()
numerical_colms2 = ['Dependents','Loan_Amount_Term']
df_test[numerical_colms2] = min_scaler.fit_transform(df_test[numerical_colms2])
df_test


# In[180]:


# standard scaling for features having gaussian distribution in training dataset
std_scaler = StandardScaler()
numerical_colms3 = ['ApplicantIncome','LoanAmount','CoapplicantIncome']
df_train[numerical_colms3] = std_scaler.fit_transform(df_train[numerical_colms3])
df_train


# In[182]:


# standard scaling for features having gaussian distribution in testing dataset
std_scaler = StandardScaler()
numerical_colms4 = ['ApplicantIncome','LoanAmount','CoapplicantIncome']
df_test[numerical_colms4] = std_scaler.fit_transform(df_test[numerical_colms4])
df_test


# # Using PCA for feature reduction

# In[ ]:





# # Model Training

# In[187]:


# Seperating Features and labels
X = df_train.drop(['Loan_Status'],axis =1)
y = df_train['Loan_Status']
X_test1 = df_test


# In[189]:


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 46)


# ## Naive Bayes
# Training and prediction on training data
naive_bayes_model = GaussianNB()
naive_bayes_model.fit(X_train,y_train)
y_pred = naive_bayes_model.predict(X_test)# On training data
print ("accuracy_score:" , accuracy_score(y_pred,y_test))#  prediction on testing data
y_pred1 = naive_bayes_model.predict(X_test1)
# ## Decision Tree

# In[193]:


# Training and prediction on training data
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
y_pred = decision_tree_model.predict(X_test)


# In[195]:


# On training data
print ("accuracy_score:" , accuracy_score(y_pred,y_test))


# In[197]:


#  prediction on testing data
y_pred1 = decision_tree_model.predict(X_test1)


# ## Random Forest
# Training and prediction on training data
Random_forest_model = RandomForestClassifier()
Random_forest_model.fit(X_train,y_train)
y_pred = Random_forest_model.predict(X_test)# On training data
print ("accuracy_score:" , accuracy_score(y_pred,y_test))#  prediction on testing data
y_pred1 = Random_forest_model.predict(X_test1)
# ## Logistic Regression
# Training and prediction on training data
Logisticreg_model = LogisticRegression()
Logisticreg_model.fit(X_train,y_train)
y_pred = Logisticreg_model.predict(X_test)# On training data
print ("accuracy_score:" , accuracy_score(y_pred,y_test))#  prediction on testing data
y_pred1 = Logisticreg_model.predict(X_test1)
# ## SVC
# Training and prediction on training data
SVC_model = SVC()
SVC_model.fit(X_train,y_train)
y_pred = SVC_model.predict(X_test)# On training data
print ("accuracy_score:" , accuracy_score(y_pred,y_test))#  prediction on testing data
y_pred1 = SVC_model.predict(X_test1)
# # Saving Results

# In[203]:


result_df = pd.DataFrame(y_pred1)
result_df


# In[205]:


result_df = pd.concat([Loan_ID,result_df],axis = 1)
result_df.rename(columns={0 : 'Loan_Status'}, inplace=True)


# In[206]:


result_df


# In[217]:


result_df.to_csv("submission_decisiontree.csv",index=False)


# In[219]:


saved_file = pd.read_csv('submission_decisiontree.csv')
print(saved_file.head())


# In[ ]:




