import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('healthcare-dataset-stroke-data.csv')
#print(data)
#print(data.isnull().sum())
#print(data.describe())
'''print(data.dtypes)'''

#............... drop unused column in the prediction...............
data = data.drop('id', axis=1)
data = data[data['gender']!='Other']
print(data['gender'].unique())

#............... fill the null values in the columns ...............

'''print("Mean", data['bmi'].mean())   >>> to get the mean of bmi column
print("Median", data['bmi']. median())  >>> to get the median of bmi column (if there no outliers) so will us mean
'''
data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

#............... Encoding(object) for the used columns ...............
'''print(data['gender'].unique())
print(data['gender'].mode())
print(data['ever_married'].unique())
print(data['work_type'].unique())
print(data['Residence_type'].unique())
print(data['smoking_status'].unique())'''

data['gender'] = data['gender'].map({'Male':0, 'Female':1})
data['ever_married'] = data['ever_married'].map({'Yes':0, 'No':1})
data['work_type'] = data['work_type'].map({'Never_worked':0, 'children':1, 'Self-employed':2, 'Govt_job':3, 'Private':4})
data['Residence_type'] = data['Residence_type'].map({'Urban':0, 'Rural':1})
data['smoking_status'] = data['smoking_status'].map({'never smoked':0, 'Unknown':1, 'formerly smoked':2, 'smokes':3})


#............... will identify input(x) and output(y) and drop O/P column from I/P ...............
x= data.drop('stroke', axis=1)
y= data['stroke']
#print(x.shape,y.shape)

#............... train_test_split ...............
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
# >>> Test that train working well <<<
'''print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)'''

#............... try the model(Training) ...............
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)

'''scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

log = LogisticRegression(max_iter=3000, class_weight='balanced', C=10) #notified that model output all not stroke as the file includes small count of stroke people so added class weight
log.fit(x_train_scaled, y_train)
y_pred = log.predict(x_test_scaled)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
print("Accuracy = ", acc) #74%
print("Report", classification_report(y_test, y_pred))'''

log = LogisticRegression(max_iter=1000, class_weight='balanced', C=10) #notified that model output all not stroke as the file includes small count of stroke people so added class weight
log.fit(x_train, y_train)
y_pred = log.predict(x_test)
print(y_pred)
acc = accuracy_score(y_test, y_pred)
print("Accuracy = ", acc) #74%
print("Report", classification_report(y_test, y_pred))


#............... to deploy the model ...............
with open('healthCareStrokeCheck.pkl', "wb") as file:
    pickle.dump(log, file)
print("model saved")
print(x)



