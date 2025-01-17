# -*- coding: utf-8 -*-
"""RNN-fault classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BGsy7FLboC7RpGNh3NGuZpF2X0hDM8Sp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import scipy.io



mat0 = scipy.io.loadmat('outputdata0-new.mat')
mat1 = scipy.io.loadmat('outputdata1-new.mat')
mat2 = scipy.io.loadmat('outputdata2-new.mat')
mat3 = scipy.io.loadmat('outputdata3-new.mat')
mat4 = scipy.io.loadmat('outputdata4-new.mat')
mat5 = scipy.io.loadmat('outputdata5-new.mat')
mat6 = scipy.io.loadmat('outputdata6-new.mat')


#  Importing the training set
# dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

X_input1 = mat0['data0_new']

X_input2 = mat1['data1_new']

X_input3 = mat2['data2_new']

X_input4 = mat3['data3_new']

X_input5 = mat4['data4_new']

X_input6 = mat5['data5_new']

X_input7 = mat6['data6_new']



    
X_input=np.zeros((16000,11,7),dtype=float)
X_input[:,:,0]=X_input1
X_input[:,:,1]=X_input2
X_input[:,:,2]=X_input3
X_input[:,:,3]=X_input4
X_input[:,:,4]=X_input5
X_input[:,:,5]=X_input6
X_input[:,:,6]=X_input7
output_label=X_input[:,8,0]
X_input=X_input[:,0:6,:]
gen_rated_current=2400
X_input=X_input/gen_rated_current

## Put all the parameters of X in one Column
# X_column=X_input.flatten()


# Feature Scaling
#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range = (0, 1))
#training_set_scaled = sc.fit_transform(training_set)

#test=np.append(X_input, X_input[0:1,0:3], axis=0)
#X_input=np.transpose(X_input)
# Creating a data structure with 50 timesteps and 1 output
#X2D=np.zeros((1,50),dtype=float)

RNNsample=5
X=[]
X3Dminor=[]
X3Dmajor= []



X3Dmajor=np.zeros((X_input.shape[0]-RNNsample,RNNsample,6),dtype=float)
X3Dminor=np.zeros((X_input.shape[0]-RNNsample,RNNsample,6),dtype=float)


for k in range (1, X_input.shape[2]+1):
    for j in range (1, X_input.shape[1]+1):
        X2D=[]       
        for i in range(RNNsample, X_input.shape[0]):
            X2D.append(X_input[i-RNNsample:i,j-1,k-1])
            #X2D=np.append(X2D,X_input[j-1:j,i-50:i], axis=0)
    
        X2D = np.array(X2D)  
        X3Dminor[:,:,j-1]=X2D 
    X3Dmajor=np.append(X3Dmajor,X3Dminor, axis=0)
    
    

X=X3Dmajor[(X_input.shape[0]-RNNsample):,:,:]

""" Creating the output label. Be careful! The output label is designed based on 7 series of simulations. Thus, the output_label repeated 7 times!"""

output_label=output_label[5:]
y=output_label

for i in range (0,6):
    y=np.append(y,output_label,0)
    

"""y=np.zeros((len(X),1),dtype=int)
y[5951:(X_input.shape[0]-RNNsample)]=1
y[5951+(X_input.shape[0]-RNNsample):2*(X_input.shape[0]-RNNsample)]=2
y[5951+2*(X_input.shape[0]-RNNsample):3*(X_input.shape[0]-RNNsample)]=3
y[5951+3*(X_input.shape[0]-RNNsample):4*(X_input.shape[0]-RNNsample)]=4
y[5951+4*(X_input.shape[0]-RNNsample):5*(X_input.shape[0]-RNNsample)]=5
y[5951+5*(X_input.shape[0]-RNNsample):6*(X_input.shape[0]-RNNsample)]=6
y[5951+6*(X_input.shape[0]-RNNsample):7*(X_input.shape[0]-RNNsample)]=7
y[5951+7*(X_input.shape[0]-RNNsample):8*(X_input.shape[0]-RNNsample)]=8
y[5951+8*(X_input.shape[0]-RNNsample):9*(X_input.shape[0]-RNNsample)]=9
"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from keras.utils import np_utils
y_test = np_utils.to_categorical(y_test)
y_train = np_utils.to_categorical(y_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 6)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 12, activation='softmax'))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 128)

y_pred = regressor.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred=np.argmax(y_pred, axis = 1)
y_test=np.argmax(y_test, axis = 1)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

accuracy_score(y_test, y_pred)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (16,8))
f = sns.heatmap(cm, annot=True)
