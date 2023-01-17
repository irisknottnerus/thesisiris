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


# In[2]:


df = pd.read_csv('Desktop/Thesis/export 3/usage/alldata1.csv')


# In[3]:


df.head()


# In[ ]:


df = pd.read_csv('Desktop/Thesis/export 3/usage/elecdata.csv')


# In[ ]:


df2 = pd.read_csv('Desktop/Thesis/export 3/usage/datalos.csv')
df2.head()


# In[4]:


#Formating to DateTime
df['date'] = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%Sz') for x in df['date']]


# In[5]:


df = df.groupby('date', as_index = False)['usage'].mean(0)


# In[ ]:


#Formating to DateTime
weer['date'] = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%Sz') for x in weer['date']]


# In[6]:


df.sort_values('date', inplace = True)


# In[7]:


df.head()


# df_row = pd.concat([df, df2], axis=1)
# df_row
# df_row.info()

# In[ ]:


df3 = pd.merge(df, df2, on='date')

df3.info()


# In[8]:


# Selecting all rows till a new day starts at 00:00
df = df.iloc[13: , :]


# In[9]:


df.info()
df.describe().T


# In[ ]:


del df3['usage']
del df3['Unnamed: 0']


# In[10]:


#transforming DateTime column into index
df = df.set_index('date')
df.index = pd.to_datetime(df.index)


# In[11]:


df.head()


# # Split data 
# Set Seed first

# In[12]:


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

# In[ ]:


df["usage"][:11101].plot(figsize=(8,6),legend=True, color = 'royalblue')
df["usage"][11101:].plot(figsize=(8,6),legend=True, color = 'orange')
plt.legend(['Training data','Test data '])
plt.xlabel('Date per hour', size = 16)
plt.ylabel('Energy Consummption', size = 16)
plt.title('Distribution of Training-, Test set', size = 18, weight = 'bold')
plt.show()
plt.savefig('Desktop/Thesis/export 3/usage/fotos/trainsplit.png')   # save the figure to file


# In[ ]:





# In[ ]:



df["usage"][:11101].plot(figsize=(8,6),legend=True, color = 'royalblue')
df["usage"][8881:11101].plot(figsize=(8,6),legend=True, color = 'limegreen')
df["usage"][11101:].plot(figsize=(8,6),legend=True, color = 'orange')
plt.legend(['Training data', 'Validation data', 'Test data '])
plt.ylabel('Energy Consummption', size = 16)
plt.xlabel('Date per hour', size = 16)
plt.title('Distribution of Train-, Validation & Test set', size = 18, weight = 'bold')
plt.show()
plt.savefig('Desktop/Thesis/export 3/usage/fotos/valisplit.png')   # save the figure to file


# # @Tensferflow
# 

# In[13]:


column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.60)]
val_df = df[int(n*0.60):int(n*0.75)]
test_df = df[int(n*0.75):]

num_features = df.shape[1]


# In[ ]:


print(train_df.shape)
print(val_df.shape)
print(test_df.shape)


# In[ ]:


train_df3= train_df3.rename(columns={'usage.x': 'elec', 'usage.y': 'gas' })


# In[ ]:


model = VAR(train_df3)


# In[ ]:


estimation_results = model.fit(24)
# Compute output summary of estimates
estimation_results.summary()


# In[ ]:


#lag order
lag_order = estimation_results.k_ar
# desired number of steps ahead
predictions = estimation_results    .forecast(train_df3.values[-lag_order:],              24)
# Converts NumPy into Pandas DataFrame
predictionsDF = pd.DataFrame(predictions)
# Assign column headers
predictionsDF.columns =     ['elec',      'gas']
ukSalesPredDF = predictionsDF['elec']
ukSalesPredDF


# In[ ]:


plt.figure(figsize=(14, 8))
# Plotting Actuals
plt.plot(train_df3.index, train_df3.elec, label='Actuals')
# Plotting Forecasts
plt.plot(train_df3.index, train_df3, label='Forecasts')
plt.legend(loc='best')
plt.show()


# In[15]:


train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std


# In[16]:


df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=0)


# In[18]:


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
    
    #@tenserflow


# In[19]:


w1 = WindowGenerator(input_width=1, label_width=1, shift=1,
                     label_columns=['usage'])
w1


# In[ ]:


train_df3.head()


# In[ ]:


w2 = WindowGenerator(input_width=24, label_width=1, shift=1,
                     label_columns=['usage.x'])
w2


# In[ ]:


q2 = WindowGenerator(input_width=24, label_width=1, shift=2,
                     label_columns=['usage'])
w3


# In[ ]:


q24 = WindowGenerator(input_width=24, label_width=1, shift=24,
                     label_columns=['usage'])
q24


# In[ ]:


q168 = WindowGenerator(input_width=24, label_width=1, shift=168,
                     label_columns=['usage'])
q168


# In[20]:


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


# In[21]:



example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                           np.array(train_df[100:100+w2.total_window_size]),
                           np.array(train_df[200:200+w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')


# In[ ]:


w2.example = example_inputs, example_labels


# In[74]:


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
            plt.legend() and plt.title('Visualization GRU window5 (forecast 24 hours)', weight = 'bold', size = 18)
    
    plt.xlabel('Time [h]')
    
WindowGenerator.plot = plot


# In[ ]:


w2.plot()


# In[24]:


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


# In[50]:


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


# In[28]:


# Each element is an (inputs, label) pair.
w1.train.element_spec


# In[ ]:


for example_inputs, example_labels in w1.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


# In[51]:


#model 1 hour into future
single_step_window = WindowGenerator(
    input_width=24, label_width=1, shift=1,
    label_columns=['usage'])
single_step_window


# In[ ]:


for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


# In[41]:


single_step_window = WindowGenerator(
   
    input_width=1, label_width=1, shift=1)

wide_window = WindowGenerator(
    input_width=24, label_width=1, shift=1)

for example_inputs, example_labels in wide_window.train.take(1):
  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
  print(f'Labels shape (batch, time, features): {example_labels.shape}')


# In[ ]:


var = VAR()
var.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])


# In[ ]:


val_performance = {}
performance = {}
val_performance['VAR'] = var.evaluate(wide_window.val)
performance['VAR'] = var.evaluate(wide_window.test, verbose=0)


# In[ ]:


varr = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])


# In[ ]:


varr = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=num_features)
])


# In[ ]:


history = compile_and_fit(varr, single_step_window)

IPython.display.clear_output()
val_performance['VAR24'] = varr.evaluate(single_step_window.val)
performance['VAR24'] = varr.evaluate(single_step_window.test, verbose=0)


# In[ ]:


wide_window.plot(varr)
plt.savefig('Desktop/Thesis/export 3/usage/fotos/varq2.png') 


# In[ ]:





# In[ ]:


performance['Var']


# In[ ]:


x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [T (degC), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()


# In[ ]:


for name, value in performance.items():
  print(f'{name:15s}: {value[1]:0.4f}')


# In[ ]:


modelvar = Var(train_df3, 2)

modelvar.compile(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Var'] = baseline.evaluate(single_step_window.val)
performance['Var'] = baseline.evaluate(single_step_window.test, verbose=0)


# In[ ]:


varr = VAR(wide_window.train)
varr.fit(loss=tf.keras.losses.MeanSquaredError(),
                 metrics=[tf.keras.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}
val_performance['Var'] = varr.evaluate(wide_window.val)
performance['Var'] = varr.evaluate(wide_window.test, verbose=0)


# In[ ]:


train_df.head()


# In[ ]:


# fit model
model = VAR(train_df3)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)


# In[ ]:


n = len(df3)
train_df3 = df3[0:int(n*0.60)]
val_df3 = df3[int(n*0.60):int(n*0.75)]
test_df3 = df3[int(n*0.75):]


# In[ ]:


from statsmodels.tsa.api import VAR

score_mae = []
score_mse
N_SPLITS = 3

features = ['usage.x', 'usage.y' ]
for fold, valid_quarter_id in enumerate(range(2, N_SPLITS)):
    # Fit model with (VAR)
    model = VAR(pd.concat([y_train, X_train], axis=1))
    model_fit = model.fit()
    
    # Prediction with (VAR)
    y_valid_pred = model_fit.forecast(model_fit.y, steps=len(X_valid))
    y_valid_pred = pd.Series(y_valid_pred[:, 0])

    # Calcuate metrics
    score_mae.append(mean_absolute_error(y_valid, y_valid_pred))
    score_rsme.append(math.sqrt(mean_squared_error(y_valid, y_valid_pred)))

# Fit model(VAR)
model = VAR(pd.concat([y, X[features]], axis=1))
model_fit = model.fit()

# Prediction(VAR)
y_pred = model_fit.forecast(model_fit.y, steps=len(X_valid))
y_pred = pd.Series(y_pred[:, 0])

plot_approach_evaluation(y_pred, score_mae, score_rsme, 'Vector Auto Regression (VAR)')


# In[ ]:





# baseline = Baseline(label_index=column_indices['usage'])
# 
# baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
#                  metrics=[tf.keras.metrics.MeanAbsoluteError()])
# 
# val_performance = {}
# performance = {}
# val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
# performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
# 
# 

# In[ ]:


# VAR example
from statsmodels.tsa.vector_ar.var_model import VAR
from random import random
model = VAR(data)
model_fit = model.fit()
# make prediction
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)

df.tail()


# In[ ]:


from statsmodels.tsa.ar_model import AutoReg


# In[44]:


wide_window = WindowGenerator(
    input_width=24, label_width=1, shift=1,
    label_columns=['usage'])

wide_window


# In[ ]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', baseline(wide_window.example[0]).shape)


# In[31]:


val_performance = {}
performance = {}


# In[ ]:





# In[47]:


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


# In[47]:


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





# In[48]:


CONV_WIDTH = 24
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['usage'])

conv_window


# In[49]:


conv_window.plot()


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

# In[34]:


import sys
import time
import IPython
from IPython.display import clear_output
for i in range(10):
    clear_output()
    print("Hello World!")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # lstm

# In[59]:


lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(16, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


# In[45]:


gru_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.GRU(16, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])


# In[37]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', gru_model(wide_window.example[0]).shape)


# In[ ]:


print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)


# In[ ]:


from IPython.display import clear_output
from IPython import parallel
import IPython


# In[60]:


history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)


# In[ ]:


predict = lstm_model.predict(wide_window.test, verbose  = 0)


# In[46]:


history = compile_and_fit(gru_model, wide_window)


val_performance['GRU'] = gru_model.evaluate(wide_window.val)
performance['GRU'] = gru_model.evaluate(wide_window.test, verbose=0)


# In[62]:


wide_window.plot(gru_model)


# In[ ]:


wide_window.plot(lstm_model)


# In[ ]:





# In[ ]:





# In[63]:



performance['LSTM']


# In[53]:



performance['GRU']


# In[ ]:





# In[65]:


x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')

val_mae1 = [v[metric_index] for v in val_performance.values()]
test_mae1 = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [Usage]')
plt.title('MAE validation and test data predicting one hour', size = 14, weight = 'bold')

plt.bar(x - 0.17, val_mae1, width, label='Validation', color = 'royalblue')
plt.bar(x + 0.17, test_mae1, width, label='Test', color = 'orange')

plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()


# In[56]:


x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = gru_model.metrics_names.index('mean_absolute_error')

val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [Usage]')
plt.bar(x - 0.17, val_mae, width, label='Validation', color = 'royalblue')
plt.bar(x + 0.17, test_mae, width, label='Test', color = 'orange')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()


# In[57]:


for name, value in performance.items():
    print(f'{name:12s}: {value[1]:0.4f}')


# ### def plot_loss (history, model_name):
#     plt.figure(figsize = (10, 6))
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('Model Train vs Validation Loss for ' + model_name)
#     plt.ylabel('Loss')
#     plt.xlabel('epoch')
#     plt.legend(['Train loss', 'Validation loss'], loc='upper right')
# 
# plot_loss (history, 'LSTM')

# In[ ]:


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

# In[61]:


multi_val_performance = {}
multi_performance = {}


# A recurrent model can learn to use a long history of inputs, if it's relevant to the predictions the model is making. Here the model will accumulate internal state for 24 hours, before making a single prediction for the next 24 hours.
# 
# In this single-shot format, the LSTM only needs to produce an output at the last time step, so set return_sequences=False in tf.keras.layers.LSTM.

# In[66]:


OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()
multi_window


# In[68]:


MAX_EPOCHS = 100

def compile_and_fit(model, window, patience=7):
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


# class MultiStepLastBaseline(tf.keras.Model):
#     def call(self, inputs):
#         return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])
# 
# last_baseline = MultiStepLastBaseline()
# last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
#                       metrics=[tf.keras.metrics.MeanAbsoluteError()])
# 
# multi_val_performance = {}
# multi_performance = {}
# 
# multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
# multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)
# multi_window.plot(last_baseline)

# In[ ]:


multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(16, return_sequences=False),
    # Shape => [batch, out_steps*features].
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
plt.savefig('Desktop/Thesis/export 3/usage/fotos/pred168.png')   


# In[73]:


multi_window.plot(multi_lstm_model)


# In[ ]:





# In[ ]:


test_df.head()


# In[70]:


multi_gru_model = tf.keras.Sequential([
    tf.keras.layers.GRU(16, return_sequences=False),
 
    # hidden layer
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),

    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history_gru = compile_and_fit(multi_gru_model, multi_window)

IPython.display.clear_output()

multi_val_performance['GRU'] = multi_gru_model.evaluate(multi_window.val)
multi_performance['GRU'] = multi_gru_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_gru_model)


# In[75]:


multi_window.plot(multi_gru_model)


# In[76]:


multi_performance['LSTM']


# In[ ]:


multi_performance['LSTM']


# In[77]:


multi_performance['GRU']


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
plt.title('MAE predicting 168 hours', size = 17,weight = 'bold')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()


# In[ ]:


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


for name, value in multi_performance.items():
  print(f'{name:8s}: {value[1]:0.4f}')

