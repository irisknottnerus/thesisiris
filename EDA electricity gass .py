#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import concat
from pandas import DataFrame
from pandas.plotting import lag_plot
from pandas.plotting import parallel_coordinates

import matplotlib.pyplot as plt
from matplotlib import pyplot

import numpy as np
import seaborn as sns
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller

# visualization import
get_ipython().run_line_magic('matplotlib', 'inline')

# define the plot size default
from pylab import rcParams
rcParams['figure.figsize'] = (12,5)

# load specific forecasting tools
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import mse,rmse

# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


weer = pd.read_csv('Desktop/Thesis/export 3/usage/elecweer.csv')


# In[3]:


weer['date'] = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%Sz') for x in weer['date']]


# advanced timeseries
# https://www.kaggle.com/code/davidanimaddo/energy-timeseries-advanced-data-visualization#1.-Introduction

# In[12]:


df = pd.read_csv('Desktop/Thesis/export 3/usage/datalos1.csv')


# In[5]:


weer['date'] = pd.to_datetime(weer['date'])


# In[13]:


df['date'] = pd.to_datetime(df['date'])


# In[14]:


df = df.set_index('date')


# In[139]:


weer = weer.set_index('date')


# In[140]:


# Select the proper time period for hourly aggreagation
df = df['2020-11-26 00:00:00':'2022-09-22 00:00:00'].resample('H').sum()
df.head()


# In[141]:


# Select the proper time period for hourly aggreagation
weer = weer['2020-11-26 00:00:00':'2022-09-22 00:00:00'].resample('H').sum()
weer.head()


# In[142]:


weer.info()
weer.isna()


# #Formating to Date Time
# df['date'] = [datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in df['date']]

# In[143]:


#Check for missing values
print('Total num of missing values:') 
print(weer.isna().sum())
print('')
weer_missing_date = weer.loc[weer.isna() == True]


# In[7]:


#Check for missing values
print('Total num of missing values:') 
print(df.gass.isna().sum())
print('')
df_missing_date = df.loc[df.gass.isna() == True]


# In[209]:


weer.head()
df = weer.iloc[:, 0:5,]
print(df.head())


# In[17]:


df = df.iloc[13: , :]
df.head()


# In[16]:


del df['Unnamed: 0']


# In[18]:


df.info()
df.describe().T


# df.index = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:&s')

# df.index = pd.DatetimeIndex(df.index).to_period('H')

# In[ ]:


df.head()


# In[151]:


print(df.index.freq)


# del df['date']

# In[ ]:


df.head()


# In[ ]:


# Visualize data
Date = df['date']
sns.set(rc={'figure.figsize':(9,7)})
sns.lineplot(x=Date, y=df['elec'], alpha = 0.9)
plt.legend(['elec'])
plt.title('Time Series after cleaning - Electricity Dataset', size= 18, weight = 'bold')
plt.xlabel('Date per hour', size = 16)
plt.ylabel(' Electricity Consumption', size = 16)
plt.savefig('Desktop/Thesis/export 3/usage/elecline.png')   # save the figure to file



# In[ ]:


# Visualize data
Date = df['date']
sns.set(rc={'figure.figsize':(9,7)})
sns.lineplot(x=Date, y=df['gass'], alpha = 0.9)
plt.legend(['gass'])
plt.title('Time Series after cleaning - Gass Dataset', size= 18, weight = 'bold')
plt.xlabel('Date per hour', size = 16)
plt.ylabel(' Gass Consumption', size = 16)
plt.savefig('Desktop/Thesis/export 3/usage/gassline.png')   # save the figure to file



# In[39]:


weer.isna().sum()


# ### Exploratory Data Analysis (EDA)

# In[ ]:


#transforming DateTime column into index
df2 = df.set_index('date')
df2.index = pd.to_datetime(df2.index)


# Running the example plots the energy data (t) on the x-axis against the energy on the previous day (t-1) on the y-axis.

# In[ ]:


values = DataFrame(df.values)
dataframe = concat([values.shift(336), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[ ]:


lag_plot(df)
pyplot.show()


# # Split data

# In[ ]:


from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# temp = df.copy() #Temporary copy 
# 
# dataset = df.astype('float32') 
# dataset = np.reshape(dataset, (-1, 2))
# scaler = MinMaxScaler(feature_range=(0, 1)) # Min Max scaler
# dataset = scaler.fit_transform(dataset) # fit and transform the dataset
# 
# train_size = int(len(dataset) * 0.80) 
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# #new https://www.kaggle.com/code/varanr/hourly-energy-demand-time-series-forecast#Preprocessing
# print(train.shape)
# print(test.shape )

# ### Testing Stationarity
# Since the VAR model requires the time series you want to forecast to be stationary, it is customary to check all the time series in the system for stationarity.https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# Augmented Dickey-Fuller test An augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. It is basically Dickey-Fuller test with more lagged changes on RHS. https://analyticsindiamag.com/complete-guide-to-dickey-fuller-test-in-time-series-analysis/
# 

# In[40]:


def dickey_fuller(series,title='Your Dataset'):
    '''Hypothesis Test for stationarity '''
    print(f'Augmented Dickey Fuller Test for the dataset {title}')
    
    result = adfuller(series.dropna(),autolag='AIC')
    labels = ['ADF test statistics','p-value','#lags','#observations'] # use help(adfuller) to understand why these labels are chosen
    
    outcome = pd.Series(result[0:4],index=labels)
    
    for key,val in result[4].items():
        outcome[f'critical value ({key})'] = val
        
    print(outcome.to_string()) # this will not print the line 'dtype:float64'
    
    if result[1] <= 0.05:
        print('Strong evidence against the null hypothesis') # Ho is Data is not stationary, check help(adfuller)
        print('Reject the null hypothesis')
        print('Data is Stationary')
    else:
        print('Weak evidence against the Null hypothesis')
        print('Fail to reject the null hypothesis')
        print('Data has a unit root and is non stationary')
        


# In[45]:


dickey_fuller(df['elec'],title='elec')


# In[44]:


dickey_fuller(df['gass'],title='gass')


# In[59]:


df[10032:10034]


# In[ ]:





# In[ ]:





# def var(X, pred_step):
#     N, T = X.shape
#     temp1 = np.zeros((N, N))
#     temp2 = np.zeros((N, N))
#     for t in range(1, T):
#         temp1 += np.outer(X[:, t], X[:, t - 1])
#         temp2 += np.outer(X[:, t - 1], X[:, t - 1])
#     A = temp1 @ np.linalg.inv(temp2)
#     mat = np.append(X, np.zeros((N, pred_step)), axis = 1)
#     for s in range(pred_step):
#         mat[:, T + s] = A @ mat[:, T + s - 1]
#     return mat[:, - pred_step :]

# pred_step = 2
# mat_hat = var(df, pred_step)
# print(mat_hat)

# In[ ]:





# In[19]:


# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic


# In[ ]:


# Plot
fig, axes = plt.subplots(nrows=2, ncols=2, dpi=100
                         , figsize=(10,6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# In[ ]:


from statsmodels.tsa.vector_ar.vecm import coint_johansen

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(df)


# In[ ]:


model = VAR(df)


# In[ ]:


x = model.select_order(maxlags=48)
x.summary()


# In[ ]:


model_fitted = model.fit(24)
model_fitted.summary()


# In[ ]:


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue'] 
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting h0.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the h0.")
        print(f" => Series is Non-Stationary.") 


# In[ ]:


# ADF Test on each column
for name, column in df.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')


# In[ ]:


from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(adjust(col), ':', round(val, 2))


# In[ ]:


# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df.values[-lag_order:]
forecast_input


# In[ ]:


# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=24)
df_forecast = pd.DataFrame(fc, index=df.index[-24:], columns=df.columns + '_2d')
df_forecast


# In[ ]:


fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# In[ ]:





# In[152]:


from statsmodels.tsa.stattools import grangercausalitytests


# In[153]:


nobs = 5929
train = df[:-nobs]
test = df[-nobs:]


# In[154]:


len(train), len(test)


# In[155]:


p = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]  # try with list of 7 p values

for i in p:
    model = VAR(train)
    results = model.fit(i)
    print(f'VAR Order {i}')
    print('AIC {}'.format(results.aic))
    print('BIC {}'.format(results.bic))
    print()


# In[156]:


# lets confirm that both the variables are included in the model
model.endog_names


# In[157]:


results = model.fit(5)
results.summary()


# Predict the next 12 values
# For predictions, VAR model uses .forecast() instead of predictions. This is similar to theHolt Winters. One of the requirement for VAR model is that we need to pass the lag order of the number of previous observations as well. Unfortunately, this lag order does not have the datetime index and hence we will have to build this ourselves.

# In[87]:


lag_order = results.k_ar
lag_order


# In[89]:


z = results.forecast(y=train.values[-lag_order:],steps = 12)
z


# In[94]:


idx = pd.date_range(start='2022-01-18 00:00:00',periods=12,freq='H')
df_forecast = pd.DataFrame(z,index=idx,columns=['ELEC2D','GASS2D'])


# In[95]:


df_forecast[:5]


# In[ ]:


df_forecast['ELEC1D'] = (df['elec'].iloc[-nobs-1] - df['elec'].iloc[-nobs-2]) + df_forecast['ELEC2D'].cumsum()


# In[ ]:


test


# In[ ]:


# Now build the forecast values from the first difference set
df_forecast['ELECFORE'] = df['elec'].iloc[-nobs-1]+ df_forecast['ELEC2D'].cumsum()
df_forecast


# In[96]:


test


# In[97]:


df_forecast


# In[98]:


results.plot()


# In[99]:


results.plot_forecast(12)


# In[101]:


df['elec'][-nobs:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
df_forecast['ELEC2D'].plot(legend=True);


# In[ ]:





# # Goeie

# https://blog.devgenius.io/implementing-vector-autoregression-from-scratch-with-python-b12eedbf35ad hieronder

# In[158]:


# Your optimal lag
optimal_lag = 24

# Extract column names
features = df.columns

# Loop through each lags
for i in range(1, optimal_lag + 1):
    
    # Loop through each features
    for j in features:
        
        # Add lag i of feature j to the dataframe
        df[f"{j}_Lag_{i}"] = df[j].shift(i)
        
# Remove all missing values
df = df.dropna()


# In[159]:


df.head()


# In[161]:



y_elec = df["usage"]
df = df.drop(["elec"], axis = 1)

# Insert an intercept column with value of 1 throughout
df.insert(0, "Intercept", 1)

# Transform dataframe to matrix
X = df.to_numpy()
y_elec = y_elec.to_numpy()
y_gass = y_gass.to_numpy()


# In[162]:


print(y_elec)


# In[163]:


def NormalEquations(X, y):

    # Normal equations
    XtX = np.matmul(X.T, X)
    XtY = np.matmul(X.T, y)
    XtX_Inv = np.linalg.inv(XtX)
    
    b = np.matmul(XtX_Inv, XtY)
    
    return b

# Obtain parameter estimates for price
b_elec = NormalEquations(X, y_elec)

# Obtain parameter estimates for demand
b_gass = NormalEquations(X, y_gass)


# In[164]:


print(b_elec)


# In[165]:


elec_estimates = []
gass_estimates = []

for i in df.index:
    
    # Take a row of data and turn into numpy
    entry = df.loc[i].to_numpy()
    
    # Find estimate of price
    elec_hat = np.dot(b_elec, entry) # dot product
    elec_estimates.append(elec_hat)
    
    # Find estimate of demand
    gass_hat = np.dot(b_gass, entry) # dot product
    gass_estimates.append(gass_hat)


# In[166]:


N = len(y_elec)
p = len(df.columns)
# We already have X, y, and b

def SumOfSquares(y, X, b):

    # Just copying the formula
    SStot = np.matmul(y.T, y)
    SSreg = np.matmul(np.matmul(y.T, X), b)
    SSres = np.matmul((y - np.matmul(X, b)).T, (y - np.matmul(X, b)))
    
    return SStot, SSreg, SSres
  
def RSquared(SSres, SStot, y, n):
    
    # Just copying the formula
    R = 1 - SSres / (SStot - sum(y) ** 2 / n)
    return R

def MSE(SSres, n):
    
    # Just copying the formula
    M = SSres/n
    return M

def Fstat(SSreg, SSres, n, p):
    
    # Just copying the formula
    F = (SSreg / p) / (SSres / (n - p))
    return F

def Diagnostics(y, X, b, n, p):
    
    # To keep things neat
    SStot, SSreg, SSres = SumOfSquares(y, X, b)
    
    R = RSquared(SSres, SStot, y, n)
    print(f"The R-squared is: {round(R, 2)}")
    
    M = MSE(SSres, n)
    print(f"The MSE is: {round(M, 2)}")
    
    F = Fstat(SSreg, SSres, n, p)
    print(f"The F-statistic is: {round(F, 2)}")


# In[167]:


Diagnostics(y_elec, X, b_elec, N, p)


# In[121]:


Diagnostics(y_gass, X, b_gass, N, p)


# In[168]:


y_elec


# In[169]:


b_elec


# In[186]:


PointsAhead = 100 # You should enter however many points you want to predict
ExPostElec = []
ExPostGass = []

data = df.iloc[-1].to_list()
for i in range(PointsAhead):
    
    # Forecast new values given current predictors
    PredictElec = np.dot(data, b_elec)
    PredictGass = np.dot(data, b_gass)

    data = data[:-2] # Remove last lag
    data.insert(1, PredictGass) # Insert forecasted demand as predictor
    data.insert(1, PredictElec) # Insert forecasted price as predictor
    
    # Store values in list
    ExPostElec.append(PredictElec)
    ExPostGass.append(PredictGass)


# In[115]:


df.head()


# In[187]:


print(ExPostElec)


# In[202]:


print(data)


# In[189]:


y_elec


# In[ ]:





# In[ ]:





# In[ ]:


def tsplot2(y, title, lags= None, figsize=(12, 8)):
    fig = plt.figure(figsize = figsize)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout,(0, 0))
    hist_ax = plt.subplot2grid(layout,(0, 1))
    acf_ax = plt.subplot2grid(layout,(1, 0))
    pacf_ax = plt.subplot2grid(layout,(1, 1))
    
    y.plot(ax = ts_ax)
    ts_ax.set_title(title, fontsize = 14, fontweight = 'bold')
    y.plot(ax = hist_ax, kind = 'hist', bins = 5)
    hist_ax.set_title('Histogram')
    plot_acf(y, lags = lags, ax = acf_ax)
    plot_pacf(y, lags = lags, ax = pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax


# In[ ]:


from statsmodels.graphics.tsaplots import plot_pacf, plot_acf #Lags


# In[ ]:


num_var = len(df.iloc[1,:])
for i in range(0, num_var):
    tsplot2(df.iloc[:,i].dropna(), title = df.columns[i], lags = 50)


# In[ ]:


nobs = 24
train = df[:-nobs]
test = df[-nobs:]


# In[ ]:


test


# # weather lstm

# In[210]:


df.head()


# In[216]:


from pandas import read_csv
from matplotlib import pyplot
# load dataset
values = df.values
# specify columns to plot
groups = [0, 1, 2, 3, 4]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(df.columns[group], y=0.4, loc='right')
	i += 1
pyplot.show()


# In[231]:


# prepare data for lstm
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

values = df.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[7,8,9]], axis=1, inplace=True)
print(reframed.head())


# In[232]:


reframed.head()


# In[233]:


...
# split into train and test sets
values = reframed.values
n_train_hours = 450 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




