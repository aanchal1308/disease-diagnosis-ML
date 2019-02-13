# Diabetes Onset Detection using neural network and grid search

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.layers import Dropout
from sklearn.metrics import classification_report, accuracy_score

# import the uci pima indians diabetes dataset
url = "http://ftp.ics.uci.edu/pub/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['n_pregnant', 'glucose_concentration', 'blood_pressuer (mm Hg)', 'skin_thickness (mm)', 'serum_insulin (mu U/ml)',
        'BMI', 'pedigree_function', 'age', 'class']
df = pd.read_csv(url, names = names)

# Describe the dataset
print(df.describe())
print(df[df['glucose_concentration'] == 0])

# Preprocess the data, mark zero values as NaN and drop
columns = ['glucose_concentration', 'blood_pressuer (mm Hg)', 'skin_thickness (mm)', 'serum_insulin (mu U/ml)', 'BMI']
for col in columns:
    df[col].replace(0, np.NaN, inplace=True)
print(df.describe())

df.dropna(inplace=True)  # Drop rows with missing values
print(df.describe())

dataset = df.values  # Convert dataframe to numpy array
print(dataset.shape)
# Split into input (X) and output (Y)
X = dataset[:,0:8]   # Array slicing, select all rows and columns from 0 to 7
Y = dataset[:,8].astype(int)
print(X.shape)
print(Y.shape)
print(Y[:5])

# Standardize the data
scaler = StandardScaler().fit(X)
X_standardized = scaler.transform(X)
data = pd.DataFrame(X_standardized)
print(data.describe())

'''
We create a Sequential model and add layers one at a time until we are happy with our network topology. The right number of inputs in the input layer is specified with the input_dim argument. Generally, you need a network large enough to capture the structure of the problem. We will use a fully-connected network structure with three layers. Fully connected layers are defined using the Dense class. 
We can specify the number of neurons in the layer as the first argument, the initialization method as kernel_initializer and specify the activation function using the activation argument. We initialize the network weights to normal for small random numbers generated from a Gaussian distribution. 
Better performance is achieved using the rectifier activation function. We use a sigmoid on the output layer to ensure our network output is between 0 and 1.
The 1st layer has 8 neurons and expects 8 input variables. The second hidden layer has 4 neurons, the output layer has 1 neuron to predict the class (onset of diabetes or not).
'''

# Building the Keras model
def create_model():
    # create sequential model
    model = Sequential()    
    model.add(Dense(8, input_dim = 8, kernel_initializer = 'normal', activation = 'relu')) #rectifier(relu) activation function
    model.add(Dense(4, input_dim = 8, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid')) #sigmoid function in output layer

    #compile the model
    ''' loss function to use to evaluate a set of weights, the optimizer used to search through different weights for the network and metrics we would like to collect and report during training.
    We will use logarithmic loss, which for a binary classification problem is defined in Keras as binary_crossentropy and the efficient gradient descent algorithm adam as it is an efficient default.'''
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy']) # metrics for classification problem: classification accuracy
    return model

model = create_model()
print(model.summary())

# optimize a network by tuning the hyperparameters.

# Performing a grid search for the optimal batch size and number of epochs

seed = 6  # Define a random seed
np.random.seed(seed)

# Start defining the model
def create_model():
    
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

#Create the model
model = KerasClassifier(build_fn = create_model, verbose = 1)

# define the grid search parameters
batch_size = [10,20,40]
epochs = [10, 50, 100]

# make a dictionary of the grid search parameters
#The training process will run for a fixed number of iterations through the dataset called epochs
#the number of instances that are evaluated before a weight update in the network is performed, called the batch size 
param_grid = dict(batch_size = batch_size, epochs = epochs)

# build and fit the GridSearchCV- exhaustive search over specified parameter values for an estimator
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state = seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# Summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
std = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, std, params):
    print('{0} ({1} with: {2}'.format(mean, stdev, param))

# Do a grid search for learning rate and dropout rate
# Reducing overfitting using dropout regularization
seed = 6
np.random.seed(seed)

# defining the model
def create_model(learn_rate, dropout_rate):    
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, input_dim = 8, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(lr = learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

# grid search parameters
learn_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.0, 0.1, 0.2]

# make a dictionary of the grid search parameters
param_grid = dict(learn_rate = learn_rate, dropout_rate = dropout_rate)

# build and fit the GridSearchCV- exhaustive search over specified parameter values for an estimator
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state = seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# Summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
std = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, std, params):
    print('{0} ({1} with: {2}'.format(mean, stdev, param))

# Do a grid search to optimize kernel initialization and activation functions
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(activation, init):
    
    model = Sequential()
    model.add(Dense(8, input_dim = 8, kernel_initializer= init, activation=activation))
    model.add(Dense(4, input_dim = 8, kernel_initializer= init, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    # We will modify the learning rate of the Adam optimizer to 0.001, as this is the best value that we found.
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# Create the model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

# Define the grid search parameters
activation = ['softmax', 'relu', 'tanh', 'linear']
init = ['uniform', 'normal', 'zero']

# make a dictionary of the grid search parameters
param_grid = dict(activation = activation, init = init)

# build and fit the GridSearchCV- exhaustive search over specified parameter values for an estimator
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state = seed), verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# Summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
std = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, std, params):
    print('{0} ({1} with: {2}'.format(mean, stdev, param))

# Do a grid search to find the optimal number of neurons in each hidden layer
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(neuron1, neuron2):
    
    model = Sequential()
    model.add(Dense(neuron1, input_dim = 8, kernel_initializer='uniform', activation='linear'))
    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer='uniform', activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(lr = 0.001)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# Create the model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

# Define the grid search parameters
neuron1 = [4, 8, 16]
neuron2 = [2, 4, 8]

# make a dictionary of the grid search parameters
param_grid = dict(neuron1=neuron1, neuron2=neuron2)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state = seed), refit=True, verbose = 10)
grid_results = grid.fit(X_standardized, Y)

# Summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
std = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, std, params):
    print('{0} ({1} with: {2}'.format(mean, stdev, param))

# generate predictions with optimal hyperparameters
y_pred = grid.predict(X_standardized)
print(y_pred.shape)
print(y_pred[:5])

print(accuracy_score(Y,y_pred))
print(classification_report(Y,y_pred))

#prediction for any example
example = df.iloc[1]
print(example)
prediction = grid.predict(X_standardized[1].reshape(1,-1))
print(prediction)
