
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import scipy.io
import imblearn
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot

mat0 = scipy.io.loadmat('outputdata0-new2.mat')
mat1 = scipy.io.loadmat('outputdata1-new2.mat')
mat2 = scipy.io.loadmat('outputdata2-new2.mat')
mat3 = scipy.io.loadmat('outputdata3-new2.mat')
mat4 = scipy.io.loadmat('outputdata4-new2.mat')
mat5 = scipy.io.loadmat('outputdata5-new2.mat')
mat6 = scipy.io.loadmat('outputdata6-new2.mat')


#  Importing the training set
# dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

X_input1 = mat0['data0_new']

X_input2 = mat1['data1_new']

X_input3 = mat2['data2_new']

X_input4 = mat3['data3_new']

X_input5 = mat4['data4_new']

X_input6 = mat5['data5_new']

X_input7 = mat6['data6_new']


oversample = SMOTE()
""" output label is the 9th column. the first 6 are Generator currents (2*3 phase) the 7th and 8 the are zonal DC currents."""
output_label=X_input1[:,8]  
"""resample data because the dataset is very imbalanced"""
X_input1, output_label_new = oversample.fit_resample(X_input1, output_label)
X_input2, output_label_new = oversample.fit_resample(X_input2, output_label)
X_input3, output_label_new = oversample.fit_resample(X_input3, output_label)
X_input4, output_label_new = oversample.fit_resample(X_input4, output_label)
X_input5, output_label_new = oversample.fit_resample(X_input5, output_label)
X_input6, output_label_new = oversample.fit_resample(X_input6, output_label)
X_input7, output_label_new = oversample.fit_resample(X_input7, output_label)




counter = Counter(output_label_new)
for k,v in counter.items():
	per = v / len(output_label_new) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()


X_input=np.zeros((X_input1.shape[0],X_input1.shape[1],7),dtype=float)
X_input[:,:,0]=X_input1
X_input[:,:,1]=X_input2
X_input[:,:,2]=X_input3
X_input[:,:,3]=X_input4
X_input[:,:,4]=X_input5
X_input[:,:,5]=X_input6
X_input[:,:,6]=X_input7
X_input=X_input[:,0:6,:]
gen_rated_current=2400
X_input=X_input/gen_rated_current




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_input, output_label_new, test_size = 0.2, random_state = 0)

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

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2])))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))

# Adding a second convolutional layer
#cnn.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
#cnn.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=12, activation='softmax'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(X_train, y_train, epochs = 20)


y_pred = cnn.predict(X_test)
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
plt.figure(figsize = (20,9))
f = sns.heatmap(cm, annot=True, fmt=".0f")

"""count_arr = np.bincount(y_test)
print('Total occurences of "11" in array: ', count_arr[11])"""
