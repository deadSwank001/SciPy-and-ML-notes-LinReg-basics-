#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as scipy
import sklearn as sklearn
from scipy import linalg
from sklearn import linear_model


# In[2]:


#Test has 30 questions and worth total 150 points
#true/False questions worth 4 points each
#multiple choice questions worth 9 points each

#Let x be the number of true/false questions
#Let y be the number of multiple choice questions

#(x + y = 30)
#(4x + 9y = 150)
testQuestionVariable = np.array([[1,1], [4,9]])
testQuestionValue = np.array([30, 150])


# In[3]:


linalg.solve(testQuestionVariable, testQuestionValue)


# In[4]:


#Imputation Technique
from sklearn.impute import SimpleImputer
imp_values = SimpleImputer(missing_values = np.nan, strategy = 'mean')


# In[5]:


imp_values.fit([[3, 5],[np.nan, 7],[1, 3]])
x = [[np.nan, 2], [6, np.nan], [7, 6]]


# In[6]:


print(imp_values.transform(x))


# In[7]:


#Scikit Learn Practice
#Iputation of Missing Values

#^ lines 5,6,7

# Categoricalvariable
#EXAMPLES: 6 sided Die, Demographic information of a Demographic

#2 types of encoding: Ordinal Encoding, One-Hot Encoding


# In[8]:


#Ordinal:
data = pd.DataFrame({
  'Age':[12,34,56,22,24,35],
  'Income':['Low', 'Low', 'High', 'Medium', 'Medium', 'High']
})
data

data.Income.map({'Low':1, 'Medium':2, 'High':3})


# In[9]:


#One-Hot Encoder
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from seaborn import load_dataset
#Dataset is loaded into a Pandas Data frame data
data = load_dataset('penguins')
#Instantiate a OneHotEncoder object and assign it to 'ohe'
ohe = OneHotEncoder()
#Fit and Transform the data using the fit_transform() method
transform = ohe.fit_transform(data[['island']])
#It will return the array version of the transform using the
# .toarray() method
print(transform.toarray())
# Three columns are present in the array in the binary form because
#There are three unique values in the Island column


# In[10]:


#Print One Hot encoded categories to know the
#column labels using the .categories_ attribute of
#the encoder

print(ohe.categories_)

#Add these columns as a seperate column in the Data frame

data[ohe.categories_[0]] = transform.toarray()
data


# In[11]:


#import required libs scene above^ line 1





#THIS IS NEW ML PROJECT 01 


# In[12]:


df_adv_data = pd.read_csv("C:\\Users\\swank\\OneDrive\\Desktop\\DSc2\\OSL Datasets\\Advertising.csv", index_col = 0)


# In[13]:


df_adv_data.head()


# In[14]:


#view size of dataset
df_adv_data.size


# In[15]:


#View the shape of dataset
df_adv_data.shape


# In[16]:


#view columns
df_adv_data.columns


# In[17]:


#create a feature object from the columns
X_feature = df_adv_data[['Newspaper Ad Budget ($)','Radio Ad Budget ($)','TV Ad Budget ($)']]


# In[18]:


#view the feature object
X_feature.head()


# In[19]:


#create target object from sales column which is response to dataset
Y_target = df_adv_data[['Sales ($)']]


# In[20]:


Y_target.head()
#Doesn't referrence correctly (Nevermind)


# In[21]:


#view shapes(feature/target object)
X_feature.shape


# In[22]:


Y_target.shape


# In[23]:


#split test and training data
# by default 75% training data and 25% testing data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_feature, Y_target, random_state = 1)


# In[24]:


#view shape of and test data sets for both feature and response
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[25]:


#lin regresh model [LITERALLY CREATES MODEL]
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_train, y_train)


# In[26]:


#print intercept and coefficients
print(linreg.intercept_)
print(linreg.coef_)


# In[27]:


#prediciton
y_pred = linreg.predict(x_test)
y_pred


# In[28]:


#import required libraries for calculating MSE (mean square error)
from sklearn import metrics
import numpy as np


# In[29]:


#Calculate the Mean Squared Error(MSE)
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

