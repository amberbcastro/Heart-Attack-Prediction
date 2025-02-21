#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv("C:\\Users\\SamJutes\\Downloads\\heart_attack_prediction_dataset.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


#Removing PatientID as it is not a predictor just an identifier
#Removing Country, Continet and Hemisphere as don't contribute significantly to predicting heart attack risk and could introduce unnecessary complexity to the model.
df = df.drop(['Patient ID','Country','Continent','Hemisphere'], axis=1)  


# In[7]:


#One hot encodingategorical variables, as logistic regression requires numerical inputs.
df = pd.get_dummies(df, columns=['Sex', 'Diabetes', 'Alcohol Consumption', 'Diet'], drop_first=True)


# In[8]:


#Splitting Blood Pressure into Systolic and Diastolic so it can be represnted as inetger and not string
df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(float)
df.drop('Blood Pressure', axis=1, inplace=True)  # Remove original column


# In[9]:


#Checking to see if we have a balanced dataset
sns.countplot(x='Heart Attack Risk',data=df)


# # Train Test Split and Scaling

# In[10]:


#First we separate the features from the lable sinto 2 objects: X and y


# In[11]:


X=df.drop('Heart Attack Risk',axis = 1)


# In[12]:


y=df['Heart Attack Risk']


# In[13]:


#Balancing dataset for better predictions
from imblearn.combine import SMOTETomek 


# In[14]:


smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X, y)


# In[15]:


# Check new class distribution
from collections import Counter
print("Class distribution after resampling:", Counter(y_resampled))


# In[16]:


#Now we perform the train test split, with test size of 10% and random state of 101 for replicability
#importing needed packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.1, random_state=101)


# In[18]:


#Normalizing the X train and test set feature data
#This helps logistic regression converge faster and prevents features with larger values from dominating the model.

scaler = StandardScaler()


# In[19]:


scaled_X_train = scaler.fit_transform(X_train)


# In[20]:


scaled_X_test = scaler.transform(X_test)


# # Logistic Regression Model and Cross Validation

# In[21]:


#importing necessary packages
#Using Logistic Regression along with CV to find a well-performing C value

from sklearn.linear_model import LogisticRegressionCV


# In[22]:


log_model = LogisticRegressionCV()


# In[23]:


log_model.fit(scaled_X_train,y_train)


# In[24]:


#Optimal C
log_model.C_


# In[25]:


#Coefficients
log_model.coef_


# In[26]:


#Visualizing coeffs
coeffs=pd.Series(index=X.columns, data = log_model.coef_[0])
coeffs=coeffs.sort_values()
plt.figure(figsize=(12,6))
sns.barplot(x=coeffs.index,y=coeffs.values)
plt.xticks(rotation=90)  
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Logistic Regression Coefficients")
plt.show()


# # Model Peformance Evaluation 

# In[28]:


#importing necessary packages

from sklearn.metrics import confusion_matrix, classification_report


# In[29]:


y_pred = log_model.predict(scaled_X_test)


# In[30]:


confusion_matrix(y_test,y_pred)


# In[31]:


print(classification_report(y_test,y_pred))


# In[ ]:




