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
from tensorflow.keras.layers import Autoregressive #Dense, LSTM, Dropout
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


# df = pd.read_csv('Desktop/Thesis/export 3/usage/alldata1.csv')

# In[17]:


df = pd.read_csv('Desktop/Thesis/export 3/usage/elecweer.csv')


# In[23]:


df.head()


# In[19]:


#Formating to DateTime
df['date'] = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%Sz') for x in df['date']]


# In[13]:


df = df.groupby('date', as_index = False)['usage'].mean(0)


# In[14]:


df.sort_values('date', inplace = True)


# In[20]:


df = df.rename(columns={'X.2': 'DD', 'X.3': 'FH', 'X.4': 'FF', 'X.5': 'FX', 'X.6': 'temp', 'X.9': 'sun', 'X.14': 'sight', 'X.16': 'humi','X.10': 'q', 'X.11': 'DR', 'X.2': 'RH', 'X.18': 'IX','X.19': 'M', 'X.20': 'R','X.13': 'P','X.15': 'N' })


# In[22]:


# Selecting all rows till a new day starts at 00:00
df = df.iloc[13: , :]


# In[24]:


df.info()
df.describe().T


# In[25]:


#transforming DateTime column into index
df = df.set_index('date')
df.index = pd.to_datetime(df.index)


# In[ ]:


del df['date']


# In[27]:


del df['X.21']


# In[28]:


del df['X.22']
del df['X.23']
del df['X.12']
del df['X.17']


# In[29]:


del df['RH']
del df['FH']
del df['FF']
del df['FX']
del df['q']
del df['DR']
del df['P']
del df['N']
del df['IX']
del df['M']
del df['R']


# In[30]:


df.head()


# # Split data 
# Set Seed first

# In[31]:


for i in range(5):
    random.seed(0)# Any random number in place of 0
    print(random.randint(1, 1000))


# # Some functions to help out with
# def plot_predictions(test,predicted):
#     plt.plot(test_df, color='red',label='Real Energy Consumption')
#     plt.plot(predicted, color='blue',label='Predicted Energy Consumption')
#     plt.title('Energy Consumption Prediction')
#     plt.xlabel('Time (hourly)')
#     plt.ylabel('Energy Usage')
#     plt.legend()
#     plt.show()
# 
# def return_rmse(test,predicted):
#     rmse = math.sqrt(mean_squared_error(test, predicted))
#     print("The root mean squared error is {}.".format(rmse))

# In[120]:



df["usage"][:11101].plot(figsize=(8,6),legend=True, color = 'royalblue')
df["usage"][11101:].plot(figsize=(8,6),legend=True, color = 'orange')
plt.legend(['Training data','Test data '])
plt.xlabel('Date per hour', size = 16)
plt.ylabel('Energy Consummption', size = 16)
plt.title('Distribution of Training-, Test set', size = 18, weight = 'bold')
plt.show()
plt.savefig('Desktop/Thesis/export 3/usage/fotos/trainsplit.png')   # save the figure to file


# # Weer
# 
# df["usage"][:11101].plot(figsize=(9,7),legend=True, color = 'royalblue')
# df["usage"][11101:].plot(figsize=(9,7),legend=True, color = 'orange')
# plt.legend(['Training set','Test set '])
# plt.xlabel('Date per hour', size = 16)
# plt.ylabel('Energy Consummption', size = 16)
# plt.title('Data set (train and test split)', size = 18, weight = 'bold')
# plt.show()

# In[ ]:





# In[119]:



df["usage"][:11101].plot(figsize=(8,6),legend=True, color = 'royalblue')
df["usage"][8881:11101].plot(figsize=(8,6),legend=True, color = 'limegreen')
df["usage"][11101:].plot(figsize=(8,6),legend=True, color = 'orange')
plt.legend(['Training data', 'Validation data', 'Test data '])
plt.ylabel('Energy Consummption', size = 16)
plt.xlabel('Date per hour', size = 16)
plt.title('Distribution of Train-, Validation & Test set', size = 18, weight = 'bold')
plt.show()
plt.savefig('Desktop/Thesis/export 3/usage/fotos/valisplit.png')   # save the figure to file


# In[32]:


column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.60)]
val_df = df[int(n*0.60):int(n*0.75)]
test_df = df[int(n*0.75):]

num_features = df.shape[1]


# In[33]:


print(train_df.shape)
print(val_df.shape)
print(test_df.shape)


# In[34]:


train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# In[17]:


df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=0)


# In[35]:


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
        #storing raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    # label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

    # making window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    #tenserflow


# In[38]:


w1 = WindowGenerator(input_width=24, label_width=1, shift=1,
                     label_columns=['usage'])
w1


# In[39]:


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    return inputs, labels

WindowGenerator.split_window = split_window


# In[41]:


example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                           np.array(train_df[100:100+w1.total_window_size]),
                           np.array(train_df[200:200+w1.total_window_size])])

example_inputs, example_labels = w1.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')


# In[42]:


w1.example = example_inputs, example_labels


# In[82]:


def plot(self, model=None, plot_col='usage', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs)) 
    for n in range(max_n): 
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
        
        if label_col_index is None:
            continue
            
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='green', s=64)
        
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='royalblue', s=64)
            
        if n == 0:
            plt.legend() and plt.title('LSTM weather forecast (Input=24, predict=1)', weight = 'bold', size = 18)
    
    plt.xlabel('Time [h]')
    
WindowGenerator.plot = plot


# In[44]:


w1.plot()
plt.savefig('Desktop/Thesis/export 3/usage/fotos/window24.png') 


# In[46]:


#lstm
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=16,)
    ds = ds.map(self.split_window)
    
    return ds

WindowGenerator.make_dataset = make_dataset


# In[46]:


#gru 
def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)
    ds = ds.map(self.split_window)
    
    return ds

WindowGenerator.make_dataset = make_dataset


# In[47]:


@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
    # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
    # And cache it for next time
        self._example = result
    return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example

# @tenserflow


# In[48]:


# Each element is an (inputs, label) pair.
w1.train.element_spec


# In[49]:


for example_inputs, example_labels in w1.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


# In[50]:


#model 1 hour into future
single_step_window = WindowGenerator(
    input_width=24, label_width=1, shift=1,
    label_columns=['usage'])
single_step_window


# In[51]:


for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


# AR = AutoReg(label_index=column_indices['usage'])
# 
# AR.compile(loss=tf.keras.losses.MeanSquaredError(),
#                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
# 
# val_performance = {}
# performance = {}
# val_performance['AR'] = AR.evaluate(single_step_window.val)
# performance['AR'] = AR.evaluate(single_step_window.test, verbose=0)

# In[88]:


wide_window = WindowGenerator(
    input_width=24, label_width=1, shift=1,
    label_columns=['usage'])

wide_window


# In[89]:


val_performance = {}
performance = {}


# The blue Inputs line shows the input usage at each time step.
# 
# The green Labels dots show the target prediction value. 
# The orange Predictions crosses are the model's prediction's for each output time step.

# In[ ]:





# In[ ]:





# In[90]:


#lstm
MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(lr =0.01),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history


# In[90]:


#gru
MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(lr =0.01),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history


# In[ ]:





# history = compile_and_fit(linear, single_step_window)
# 
# val_performance['Linear'] = linear.evaluate(single_step_window.val)
# performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

# wide_window.plot(linear)

# plt.bar(x = range(len(train_df.columns)),
#         height=linear.layers[0].kernel[:,0].numpy())
# axis = plt.gca()
# axis.set_xticks(range(len(train_df.columns)))
# _ = axis.set_xticklabels(train_df.columns, rotation=90)

# dense = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=1)
# ])
# 
# history = compile_and_fit(dense, single_step_window)
# 
# val_performance['Dense'] = dense.evaluate(single_step_window.val)
# performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

# In[ ]:





# In[91]:


LABEL_WIDTH = 24
CONV_WIDTH = 24
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['usage'])

conv_window


# conv_window.plot()
# plt.savefig('Desktop/Thesis/export 3/usage/fotos/lstmexampleweer.png') 

# multi_step_dense = tf.keras.Sequential([
#     # Shape: (time, features) => (time*features)
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=32, activation='relu'),
#     tf.keras.layers.Dense(units=1),
#     # Add back the time dimension.
#     # Shape: (outputs) => (1, outputs)
#     tf.keras.layers.Reshape([1, -1]),
# ])

# print('Input shape:', conv_window.example[0].shape)
# print('Output shape:', multi_step_dense(conv_window.example[0]).shape)

# print('Input shape:', wide_window.example[0].shape)
# try:
#     print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
# except Exception as e:
#     print(f'\n{type(e).__name__}:{e}')

# In[61]:


import sys
import time
import IPython
from IPython.display import clear_output
for i in range(10):
    clear_output()
    print("Hello World!")


# history = compile_and_fit(multi_step_dense, conv_window)
# 
# IPython.display.clear_output()
# val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
# performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

# conv_window.plot(multi_step_dense)

# print('Input shape:', wide_window.example[0].shape)
# try:
#     print('Output shape:', multi_step_dense(wide_window.example[0]).shape)
# except Exception as e:
#     print(f'\n{type(e).__name__}:{e}')

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # lstm

# In[95]:


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(16, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


# In[122]:


gru_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.GRU(16, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


# In[68]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', gru_model(wide_window.example[0]).shape)


# In[93]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)


# In[178]:


from IPython.display import clear_output
from IPython import parallel
import IPython


# In[96]:


history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)


# In[117]:


predict = lstm_model.predict(wide_window.test, verbose  = 0)


# In[125]:


history = compile_and_fit(gru_model, wide_window)

IPython.display.clear_output()
val_performance['GRU'] = gru_model.evaluate(wide_window.val)
performance['GRU'] = gru_model.evaluate(wide_window.test, verbose=0)


# In[123]:


wide_window.plot(gru_model)


# In[184]:


q1val = val_performance['LSTM']


# In[154]:


q1 = performance['LSTM']


# In[98]:


qweer =  performance['LSTM']
qweer


# In[126]:


performance['GRU']


# In[ ]:





# In[127]:


x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')

val_mae1 = [v[metric_index] for v in val_performance.values()]
test_mae1 = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [Usage]')

plt.bar(x - 0.17, val_mae1, width, label='Validation', color = 'royalblue')
plt.bar(x + 0.17, test_mae1, width, label='Test', color = 'orange')

plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()


# In[138]:


x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')

val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [Usage]')
plt.bar(x - 0.17, val_mae, width, label='Validation', color = 'royalblue')
plt.bar(x + 0.17, test_mae, width, label='Test', color = 'orange')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()


# In[128]:


for name, value in performance.items():
    print(f'{name:12s}: {value[1]:0.4f}')


# def plot_loss (history, model_name):
#     plt.figure(figsize = (10, 6))
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title( 'Train vs Validation Loss ' + model_name +' - Multivariate data', size = 17, weight = 'bold')
#     plt.ylabel('Loss', size = 15)
#     plt.xlabel('epoch', size = 15)
#     plt.legend(['Train loss', 'Validation loss'], loc='upper right')
# 
# plot_loss (history, 'LSTM')
# plt.savefig('Desktop/Thesis/export 3/usage/fotos/trainlossweer.png')   

# In[112]:


def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')

plot_loss (history, 'LSTM')


# In[ ]:





# # multistep deelvraag 3

#  as metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}. You can also pass a list to specify a metric or a list of metrics for each output, such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']]

# In[145]:


multi_val_performance = {}
multi_performance = {}


# A recurrent model can learn to use a long history of inputs, if it's relevant to the predictions the model is making. Here the model will accumulate internal state for 24 hours, before making a single prediction for the next 24 hours.
# 
# In this single-shot format, the LSTM only needs to produce an output at the last time step, so set return_sequences=False in tf.keras.layers.LSTM.

# In[115]:


OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
multi_window


# In[117]:


MAX_EPOCHS = 40
metrics={'output_a':'mse', 'output_b':['mse', 'mae']}
def compile_and_fit(model, window, patience=):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(lr = 0.01),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history


# In[ ]:


multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)


# In[ ]:


test_df.head()


# In[113]:


multi_gru_model = tf.keras.Sequential([
    
    tf.keras.layers.GRU(32, return_sequences=False),

    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
  
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history_gru = compile_and_fit(multi_gru_model, multi_window)

IPython.display.clear_output()

multi_val_performance['GRU'] = multi_gru_model.evaluate(multi_window.val)
multi_performance['GRU'] = multi_gru_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_gru_model)


# In[ ]:


def plot_future(prediction, model_name, y_test):
    plt.figure(figsize=(10, 6))
    range_future = len(prediction)
    plt.plot(np.arange(range_future), np.array(y_test), 
             label='Test   data')
    plt.plot(np.arange(range_future), 
             np.array(prediction),label='Prediction')
    plt.title('Test data vs prediction for ' + model_name)
    plt.legend(loc='upper left')
    plt.xlabel('Time (day)')
    plt.ylabel('Daily water consumption ($m³$/capita.day)')

plot_future(multi_performance['LSTM'] , 'LSTM', multi_val_performance['LSTM'])


# In[ ]:


x = np.arange(len(multi_performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()


# In[ ]:





# In[ ]:


def plot_loss (history, model_name):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Train vs Validation Loss for ' + model_name)
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')

plot_loss (history_gru, 'GRU')
plot_loss (history, 'LSTM')


# In[ ]:


for name, value in multi_performance.items():
  print(f'{name:8s}: {value[1]:0.4f}')

