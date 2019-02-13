# Autistic Spectrum Disorder Screening
'''The early diagnosis of neurodevelopment disorders can improve treatment 
and significantly decrease the associated healthcare costs.'''

import pandas as pd
from sklearn import model_selection
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
from sklearn.metrics import classification_report, accuracy_score

# import the dataset
file ='dataset/autism-data.csv'
names = ['A1_Score',
        'A2_Score', 
        'A3_Score',                            
        'A4_Score',                           
        'A5_Score',                            
        'A6_Score',                           
        'A7_Score',                            
        'A8_Score',                           
        'A9_Score',                            
        'A10_Score',                          
        'age',                        
        'gender ',                             
        'ethnicity',                     
        'jundice',                        
        'family_history_of_PDD',         
        'contry_of_res',                 
        'used_app_before',                
        'result',                         
        'age_desc',                
        'relation',         
        'class'
    ]
data = pd.read_csv(file, names=names)

# print the shape of the dtaframe, so we can see how many examples we have
print 'Shape of dataframe: {}'.format(data.shape)
print data.loc[0]
# display multiple patients
print data.loc[:10]
# description of the dataframe
print data.describe()

# Data Preprocessing

# drop unwanted columns
data = data.drop(['result', 'age_desc'], axis=1)
print data.loc[:10]
# create X and Y dataset for training
x = data.drop(['class'],1)
y = data['class']
print x.loc[:10]

# convert the data to categorical values -one-hot-encoded vectors
X = pd.get_dummies(x)
#print the new categorical labels
print X.columns.values
# print an example patient from the categorical data
print X.loc[1]
# covert the class data to categorical values -one-hot-encoded vectors
Y = pd.get_dummies(y)
print Y.iloc[:10]

# Split the X and Y data into training and testing datasets
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size = 0.2)
print X_train.shape
print Y_train.shape
print X_test.shape
print Y_test.shape

# Build a neural network using Keras
'''This model will be relatively simple and will only use dense (also known as fully connected) layers.
This is the most common neural network layer. The network will have one hidden layer, use an Adam optimizer,
and a categorical crossentropy loss.''' 

# define a function to build the keras model
def create_model():
    model = Sequential()
    model.add(Dense(8, input_dim=96, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    adam=Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

model = create_model()
print(model.summary())

# Training the network - fit the model to the training data
model.fit(X_train, Y_train, epochs = 50, batch_size=10, verbose =1)

# Testing and Performance metrics

# generate classification report using predictions for categorical model
predictions = model.predict_classes(X_test)
print predictions
print('Prediction Reults for Neural Network')
print(accuracy_score(Y_test[['YES']], predictions))
print(classification_report(Y_test[['YES']], predictions))

#prediction for any example
example = X_test.iloc[1]
print(example)
prediction = model.predict_classes(example.values.reshape(1,-1))
print(prediction)

