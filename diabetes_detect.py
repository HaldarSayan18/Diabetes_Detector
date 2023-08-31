import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm       #support vector machine

#Data collection and Analysis
#PIMA Daibetes Dataset
#loading the diabetes dataset to a pandas dataframe
diabetes_data = pd.read_csv('diabetes.csv')
#printing the 1st 5 rows of the dataset
diabetes_data.head()
diabetes_data.shape
diabetes_data.describe()
diabetes_data['Outcome'].value_counts()
print("diabetes_data['Outcome']")
print(diabetes_data['Outcome'].value_counts())
print()
# 0 --> non-diabetic, 1 --> Diabetic
diabetes_data.groupby('Outcome').mean()
print("diabetes_data.groupby('Outcome')")
print(diabetes_data.groupby('Outcome').mean())
print()
#separating the data and labels
x = diabetes_data.drop(columns = 'Outcome', axis = 1)
y = diabetes_data['Outcome']

#Data Standardization
scaler = StandardScaler()
scaler.fit(x)
std_data = scaler.transform(x)
#print(std_data)
x2 = std_data
y = diabetes_data['Outcome']
#print(x2)
#print(y2)

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 2)
#print(x.shape, x_train.shape, x_test.shape)

#Training the model
classifier = svm.SVC(kernel = 'linear')
#training the svm classifier
classifier.fit(x_train, y_train)

#Model Evaluation
#accuracy score on the training data
x_train_predict = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict, y_train)
print("Accuracy score of training data: ", training_data_accuracy)
print()
#accuracy score on the test data
x_test_predict = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict, y_test)
print("Accuracy score of the test data: ",test_data_accuracy)
print()

#Making predictive system
#input_data = (4, 110, 92, 0, 0, 37.6, 0.191, 30)
input_data = (0,137,40,35,168,43.1,2.288,33)
#changing the input_data to numpy array
input_data_numpy_arr = np.asarray(input_data)
#reshape the array as predicting as 1 instane
input_data_reshaped = input_data_numpy_arr.reshape(1, -1)
#standardize the input data
std1_data = scaler.transform(input_data_reshaped)
print("standardize the input data = ",std1_data)
print()

prediction = classifier.predict(std1_data)
print("prediction = ",prediction)
if(prediction[0] == 1):
  print("The person is diabetic.")
else:
  print("The person is non-diabetic.")