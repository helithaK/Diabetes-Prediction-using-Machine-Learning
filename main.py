#Importing the dependacies
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#Data collection and analysis 

  #Loading the diabetes dataset to a pandas Datafrañme

diabetes_dataset = pd.read_csv('diabetes.csv')
#print(diabetes_dataset.head())

#Number of rows and columns in this dataset
#print(diabetes_dataset.shape)

#Getting the statistical measures of the data
# print(diabetes_dataset.describe())

print(diabetes_dataset['Outcome'].value_counts())

#Non-Diabetic --> 0
#Diabetic --> 1

print(diabetes_dataset.groupby('Outcome').mean())

#Seperating the data and labels
#Axis is = to 1 if we are dropping a specific column
#Axis is = to 0 if we are dropping a specific row
X = diabetes_dataset.drop(columns = 'Outcome', axis = 1)
Y = diabetes_dataset['Outcome'] 

# print(Y)

#Data standardization
scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

# print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']

#Train test split
X_train, X_test , Y_train , Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)

print(X.shape, X_train.shape,X_test.shape)


classifier = svm.SVC(kernel = 'linear')

#Training the support vector machine classifier
classifier.fit(X_train, Y_train)


#Model Evaluation 
#Accuracy Score 
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy of the training data : ',training_data_accuracy)


#Accuracy score on test data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy of the testing data : ',testing_data_accuracy)


#Making a predictive system 
input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')