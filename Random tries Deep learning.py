#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import warnings
import numpy as np
import seaborn as sns
from datetime import datetime
warnings.filterwarnings('ignore')

from matplotlib import pyplot
import matplotlib.pyplot as plt

import pandas as pd
from pandas import concat
from pandas import DataFrame
from pandas.plotting import lag_plot
from pandas.plotting import parallel_coordinates

#Statistical Models
import statsmodels.api as sm
from pmdarima import auto_arima
from statsmodels.tsa.api import VAR
from statsmodels.tsa.base import  datetools
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf #Lags

#Performance evaluation
import sklearn.preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
get_ipython().system('pip install scalecast #for running and evaluating forecasting models')

#Deep Learning Model
import tensorflow as tf
from keras.layers import *
from tensorflow import keras
from tensorflow.keras import layers, callbacks #Sequential
from tensorflow.keras.layers import GRU #Dense, LSTM, Dropout
#from keras.layers import LSTM
#from keras.layers import Dense
#from keras.layers import Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM

#!pip install yfinance prophet
#import yfinance as yf
# Prophet model for time series forecast #from prophet import Prophet
#%matplotlib inline
#!pip install pmdarima
#from scipy import stats


# In[7]:


df = pd.read_csv('Desktop/Thesis/export 3/usage/alldata1.csv')


# In[8]:


weer = pd.read_csv('Desktop/Thesis/export 3/usage/elecweer.csv')


# In[9]:


weer['date'] = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%Sz') for x in weer['date']]


# In[10]:


#Formating to DateTime
df['date'] = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%Sz') for x in df['date']]


# In[11]:


df = df.groupby('date', as_index = False)['usage'].mean(0)


# In[12]:


df.sort_values('date', inplace = True)


# In[13]:


df.index = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:&s')


# In[14]:


df.index = pd.DatetimeIndex(df.index).to_period('H')


# In[16]:


df['date'] = pd.to_datetime(df['date'])


# In[17]:


df.head()


# In[18]:


# Selecting all rows till a new day starts at 00:00
df = df.iloc[13: , :]


# In[19]:


df.info()
df.describe().T


# In[20]:


del df['date']


# #Select the proper time period for hourly aggreagation
# df = df['2020-11-26 00:00:00':'2022-09-22 00:00:00'].resample('H').sum()
# df.head()

# In[21]:


df.info()


# In[24]:


df['check'] = df.index - df.index(1)


# # Split data 
# Set Seed first

# In[23]:


for i in range(5):
    random.seed(0)# Any random number in place of 0
    print(random.randint(1, 1000))


# In[ ]:


def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)
    
look_back = 24 # timesteps to lookback for predictions

X_train, trainY = create_dataset(train, look_back)
X_test, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
print("Shapes: \nTraining set: {}, Testing set: {}".format(X_train.shape, X_test.shape))
print("Sample from training set: \n{}".format(X_train[0]))


# In[ ]:




import numpy as np
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

def create_model():
    model = Sequential()
    model.add(Dense(12, input_shape=(8,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

seed = 7
tf.random.set_seed(seed)
# create model
model = KerasClassifier(model=create_model, verbose=0)
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X, Y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]


# In[ ]:


#supervised
def series_to_supervised(data, n_in=1, n_out=1):
    df = DataFrame(data)
    cols = list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            agg = concat(cols, axis=1)
            agg.dropna(inplace=True)
            return agg.values


# In[ ]:


# fit
def model_fit(train, config):
    return None


# In[ ]:


# forecast
def model_predict(model, history, config):
    return 0.0


# In[ ]:


# mse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# In[ ]:


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    train, test = train_test_split(data, n_test)
    model = model_fit(train, cfg)
    history = [x for x in train]
    for i in range(len(test)):
        yhat = model_predict(model, history, cfg)
        predictions.append(yhat)
        history.append(test[i])
        error = measure_rmse(test, predictions)
        print(' > %.3f' % error)
    return error


# In[ ]:


def repeat_evaluate(data, config, n_test, n_repeats=10):
 #
    key = str(config)
 
    scores = [walk_forward_validation(data, n_test, config) for _ in range(n_repeats)]

    result = mean(scores)
    print('> Model[%s] %.3f' % (key, result))
    return (key, result)


# In[ ]:


# grid search configs
def grid_search(data, cfg_list, n_test):
 # evaluate configs
    scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
 # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# In[ ]:


# fit a model
def model_fit(train, config):
    return None


# In[ ]:


# define config
cfg_list = [1, 24, 48, 168, 672]


# In[ ]:


# forecast with a pre-fit model
def model_predict(model, history, offset):
	history[-offset]


# Now that we have a robust test harness for grid searching model hyperparameters, we can use it to evaluate a suite of neural network models.

# # neural networks
