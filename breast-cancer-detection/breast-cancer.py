# Detection of breast cancer using KNN and SVM

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=names)

# Pre-process the data
df.replace('?',-99999, inplace=True)
print(df.axes)
df.drop(['id'], 1, inplace=True)

# explore the dataset and do a few visualizations
print(df.loc[10])
# Print the shape of the dataset
print(df.shape)
# Describe the dataset
print(df.describe())
# Plot histograms for each variable
df.hist(figsize = (10, 10))
plt.show()
# Create scatter plot matrix to know relationships among variables
scatter_matrix(df, figsize = (18,18))
plt.show()

# Create X and Y datasets for training
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Testing
seed = 8
scoring = 'accuracy'

# Define models to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []

for name, model in models:
       kfold = model_selection.KFold(n_splits = 10, random_state = seed)
       # Evaluate score by Cross-Validation
       cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
       results.append(cv_results)
       names.append(name)
       msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
       print(msg)

# Make predictions on validation dataset
for name, model in models:
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       print(name)
       print(accuracy_score(y_test, predictions))
       print(classification_report(y_test, predictions))

clf = SVC(gamma='auto') # create support-vector-classifier

# get accuracy score for it
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Prediction for any example
example = np.array ([[4,2,1,1,1,2,3,2,1]])

example = example.reshape(len(example), -1) #reshape to get a column vector
prediction = clf.predict(example)
#print(prediction)
if prediction==4:
       print('Malignant\n')
elif prediction==2:
       print('Benign\n')

'''If class is 4, this means that it is malignant; so this particular cell is cancerous.
A class of 2, on the other hand, means benign or healthy.'''