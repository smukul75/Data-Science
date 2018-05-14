#Multiple Linear Regression

# Importing the libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Aoiding the Dummy Variable Trap (Always remember to choose n-1)
#3 columns generated after encoder, we will not use first column
X = X[:,1:]

#----------------------------------
#Splitting Data to Training and Test
#----------------------------------

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#----------------------------------
#Feature Scaling
#----------------------------------
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Fittig Multiple Linear Regression to the Traiing Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediction the Test set results
y_pred = regressor.predict(X_test)

#find delta between real value and predicted value (test dataset)
pred_delta = y_test - y_pred