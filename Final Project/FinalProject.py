
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('install_ext https://raw.github.com/cpcloud/ipython-autotime/master/autotime.py')
get_ipython().magic('load_ext autotime')


# In[2]:

import pyprind

from IPython.display import HTML


import numpy as np
from numpy import genfromtxt

import pandas as pd
from pandas import DataFrame
from pandas import Panel

import warnings

# import sklearn as skl
# #from sklearn.preprocessing import normalize
# from sklearn.cross_validation import KFold
# from sklearn.ensemble import RandomForestClassifier as RFC
# from sklearn.tree import DecisionTreeClassifier as DTC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder

import scipy.io as sio
from scipy.spatial import distance

from scipy.misc import imread, imsave, imresize
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt

import theano

import keras
from keras.models import *
from keras.layers import *
from keras.layers.recurrent import *
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD


# In[3]:

warnings.filterwarnings('ignore')


# In[4]:

def split_data(dataset, train_fraction=0.8):
    
    train_size = (train_fraction * np.shape(dataset)[0])
    
    np.random.shuffle(dataset)
    
    columns = np.shape(dataset)[1]-1
    x = dataset[0::,0:columns]
    y = dataset[0::,columns:]
    
    x_training, x_test = x[:train_size,:], x[train_size:,:]
    
    y_training, y_test = y[:train_size, :], y[train_size:, :]
    
    return x_training, x_test, y_training, y_test


# In[5]:

Michael = pd.read_csv('Trials/Michael.csv', header=None)

Michael.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                         'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation', 'Label']

Brooke = pd.read_csv('Trials/Brooke.csv', header=None)

Brooke.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                  'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation']

Mark = pd.read_csv('Trials/Mark.csv', header=None)

Mark.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                         'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation', 'Label']

Mark.shape


# In[6]:

Test_x = pd.read_csv('Trials/results.csv', header=None)

Test_x.columns = ['Delta', 'Theta', 'Low_Alpha', 'High_Alpha', 'Low_Beta', 'High_Beta',
                         'Low_Gamma', 'High_Gamma', 'Attention', 'Meditation']

Test_x['Label'] = 1


# In[7]:

temp = Test_x.values

test_X = temp[:, 0:9]

test_Y = temp[:, 10]

test_Y


# In[8]:

temp_Y = pd.Series(test_Y)

temp_Y = pd.get_dummies(temp_Y)

test_Y = temp_Y.values

temp2 = np.zeros((len(test_Y), 2))

test_Y = np.concatenate((test_Y, temp2), axis=1)


# In[9]:

Brooke['Label'] = 3

Brooke.shape


# In[10]:

Michael = Michael[Michael['Label'] == 1]

Michael_prime = Michael[Michael['Attention'] > 0]

Michael = Michael_prime[Michael_prime['Meditation'] > 0]


Mark_prime = Mark[Mark['Attention'] > 0]

Mark = Mark_prime[Mark_prime['Meditation'] > 0]

Mark.shape


# In[11]:

Mark.head()


# In[12]:

frames = [Michael[0:1800], Mark[0:1800], Brooke[0:1800]]

data = pd.concat(frames, copy=False, ignore_index=True)

data = data.drop_duplicates()

data = data.dropna()

#data = data.values


# In[13]:

data.shape


# In[14]:

#THIS IS WHAT YOU FIX

# frames = {'Subject 1': Michael[0:1800],
#          'Subject 2': Mark[0:1800],
#          'Subject 3': Brooke[0:1800]}

# data = Panel.from_dict(frames)

# data.shape


# In[15]:

data = data.dropna()

data.shape


# In[16]:

data = data.values


# In[17]:

X = data[:, 0:9]

Y = data[:, 10]


# In[18]:

Y = pd.Series(Y)

Y = pd.get_dummies(Y)

Y = Y.values


# In[19]:

# data = np.concatenate((X, Y), axis=1)

# np.shape(data)


# In[20]:

# Y = to_categorical(Y.astype(int), 3)

# np.shape(Y)


# In[21]:

#a = Input(shape=(5400,12))
#b = Dense(3)(a)

mlp_model = Sequential()

mlp_model.add(Dense(64, input_dim=9, init='uniform'))
mlp_model.add(Activation('tanh'))
mlp_model.add(Dropout(0.5))
mlp_model.add(Dense(64, init='uniform'))
mlp_model.add(Activation('tanh'))
mlp_model.add(Dropout(0.5))
mlp_model.add(Dense(3, init='uniform'))
mlp_model.add(Activation('softmax'))

sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

mlp_model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


# In[22]:

mlp_model.fit(X, Y, batch_size=16, nb_epoch=200, verbose=0)


# In[23]:

predicted = mlp_model.predict(test_X, batch_size=20)


# In[24]:

np.set_printoptions(suppress=True)

np.mean(predicted, axis=0)


# In[25]:

#X = X.reshape((np.shape(X)[0], 1, np.shape(X)[1]))


# In[26]:

lstm_model = Sequential()

lstm_model.add(TimeDistributed(Dense(121), input_shape=(180, 5399, 11)))
#lstm_model.add(TimeDistributed(Dense(5400))
lstm_model.add(LSTM(1000, return_sequences=True))
# lstm_model.add(TimeDistributedDense(500))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(3, init='uniform', activation='softmax'))

lstm_model.compile(optimizer=sgd,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])


# In[ ]:

lstm_model.summary()


# In[ ]:

lstm_model.fit(X, Y, batch_size=32, nb_epoch=20, verbose=1)


# In[ ]:



