import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection analysis
diabetes_dataset=pd.read_csv('C:/Users/Dell/Desktop/diabetes.csv')
diabetes_dataset.head()
X=diabetes_dataset.drop(columns='Outcome',axis=1)
Y=diabetes_dataset['Outcome']
scaler=StandardScaler();
scaler.fit(X)
standardized_data=scaler.transform(X)
X=standardized_data
Y=diabetes_dataset['Outcome']
print("-----------Standardized table---------")
print(standardized_data)
#to train the data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')

#traning the suppport vector
classifier.fit(X_train,Y_train)
#model accuracy
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print("--------------------------------------------")
print('accuracy of traning is:',training_data_accuracy)
print("--------------------------------------------")
input_data=[144,899,66,203,940,248.1,0.1467,241]
input_data_as_numpy_array = np.array(input_data)
#reshape the data
input_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data=scaler.transform(input_reshaped)
prediction=classifier.predict(std_data)
print("---------------OUTCOME----------------------")
if prediction[0]==0:
    print("not diabetic person")
else:
    print("diabetic person")
print("---------------------------------------------")
