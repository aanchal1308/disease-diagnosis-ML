# Diagnosing Coronary Artery Disease using neural networks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score

# import the heart disease dataset
# It has around 303 patients collected from the Cleveland Clinic Foundation. 
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# the names will be the names of each column in our pandas DataFrame
names = ['age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
        'class']

# read the csv
cleveland = pd.read_csv(url, names=names)

# Print the shape of the dataframe
print 'Shape of DataFrame: {}'.format(cleveland.shape)
print cleveland.loc[1]
# print the last data points
print(cleveland.loc[280:])

# remove the missing data (indicated with a '?')
data = cleveland[~cleveland.isin(['?'])]
print(data.loc[280:])

# drop rows with NaN values from dataframe
data = data.dropna(axis=0)
print(data.loc[280:])

# print the shape and data type of the dataframe
print data.shape
print data.dtypes

# transform data to numeric to enable further analysis
data = data.apply(pd.to_numeric)
print(data.dtypes)

# print data characteristics using pandas in-built describe() function
print(data.describe())

# plot histograms for each variable
data.hist(figsize = (12, 12))
plt.show()

# create X and Y datasets for training
X = np.array(data.drop(['class'],1))
y = np.array(data['class'])

# split training and testing data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

#convert the data to categorical
'''the class values in this dataset contain multiple types of heart disease with values ranging from 0 (healthy) to 4 (severe heart disease).
Consequently, we will need to convert our class data to categorical labels. For example, the label 2 will become [0, 0, 1, 0, 0].'''
Y_train = to_categorical(y_train, num_classes=None)
Y_test = to_categorical(y_test, num_classes=None)
print Y_train.shape
print Y_train[:10]

'''we can begin building a neural network to solve this classification problem. Using keras, we will define a simple neural network with one hidden layer.
Since this is a categorical classification problem, we will use a softmax activation function in the final layer of our network and a categorical_crossentropy loss during our training phase.'''

# define a function to buils keras model
def create_model():
    #create
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation = 'softmax'))
    
    #compile
    adam = Adam(lr = 0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())

#fit the model to the training data
model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose = 1)

# improving the results
'''Although we achieved promising results, we still have a fairly large error.
This could be because it is very difficult to distinguish between the different severity levels of heart disease (classes 1 - 4). 
Let's simplify the problem by converting the data to a binary classification problem - heart disease or no heart disease.'''

Y_train_binary = y_train.copy()
Y_test_binary = y_test.copy()
Y_train_binary[Y_train_binary > 0] = 1
Y_test_binary[Y_test_binary > 0] = 1

print Y_train_binary[:20]

# define new kaeras model for binary classification
def create_binary_model():
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    adam = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

binary_model = create_binary_model()
print(binary_model.summary())

# fit the binary model on the training data
binary_model.fit(X_train, Y_train_binary, epochs=100, batch_size=10, verbose=1)
#By setting verbose 0, 1 or 2 you just say how do you want to see the training progress for each epoch.

# generate classification report using predictions for categorical model
categorical_pred = np.argmax(model.predict(X_test), axis=1)
# argmax returns the indices of the maximum values along an axis.
print("Results for categorical model")
print(accuracy_score(y_test, categorical_pred))
print(classification_report(y_test, categorical_pred))

# generate classification report using prediction for binary model
binary_pred = np.round(binary_model.predict(X_test)).astype(int)
# astype is used to create copy of the array, cast to a specified type(int, here)
print('Results for Binary Model')
print(accuracy_score(Y_test_binary, binary_pred))
print(classification_report(Y_test_binary, binary_pred))
