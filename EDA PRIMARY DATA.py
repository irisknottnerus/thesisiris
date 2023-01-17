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
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller

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


# advanced timeseries
# https://www.kaggle.com/code/davidanimaddo/energy-timeseries-advanced-data-visualization#1.-Introduction

# In[2]:


df = pd.read_csv('Desktop/Thesis/export 3/usage/alldata1.csv')


# In[3]:


#Formating to Date Time
df['date'] = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%Sz') for x in df['date']]


# In[4]:


#Check for missing values
print('Total num of missing values:') 
print(df.usage.isna().sum())
print('')
df_missing_date = df.loc[df.usage.isna() == True]


# In[6]:


df.head()
df.asfreq('H').info()
df.info


# In[5]:


df.index = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:&s')


# In[8]:


del df['date']


# In[6]:


# Selecting all rows till a new day starts at 00:00
df = df.iloc[13: , :]
df.head()


# In[9]:


df.tail()


# Select the proper time period for weekly aggreagation
# df = df['2020-11-26 00:00:00':'2022-09-22 00:00:00'].resample('H').sum()
# df.tail()

# In[12]:


df.info()
df.describe().T


# In[11]:


df.index = pd.DatetimeIndex(df.index).to_period('H')


# In[ ]:





# In[ ]:





# #Calculate one-period percent change = x_t/x_{t-1}
# df['change'] = df.usage.div(df.Lag_1)
# df[['usage', 'Lag_1', 'change']].head(5)

# df['diff'] = df.usage.diff()
# df[['usage', 'diff']].head(5)             
# 

# In[ ]:





# In[86]:


usage = df.loc[:,'usage']


# In[87]:


df = pd.DataFrame({
    'usage': usage,
    'Lag_1': usage.shift(1),
    'Lag_2': usage.shift(2),
    'Lag_12': usage.shift(12),
    'Lag_24': usage.shift(24),
    'Lag_48': usage.shift(48),
    'Lag_168': usage.shift(168),
    'Lag_672': usage.shift(672),
})
df.head()


# In[89]:


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3,2, figsize = (14,14))
fig.tight_layout(pad=5)

#1
ax = sns.regplot(x='Lag_1', y='usage', data=df, ci=None, label = 'Lag_1', color = 'red',
                 scatter_kws=dict(color='royalblue', alpha = 0.5), ax = ax1)
ax.set_aspect('equal')
ax.set_title('Lag Plot (Lag_1) Energy Consumption', size = 18, weight = 'bold');
ax.set_xlabel('Lag_1 (= 1 hour)', size = 16)
ax.set_ylabel('Hourly Energy Consumption', size = 16)
ax.legend(df[2:2])


#2
ax1 = sns.regplot(x='Lag_2', y='usage', data=df, ci=None, color = 'orange', 
                 scatter_kws=dict(color='royalblue', alpha = 0.5), ax = ax2)
ax1.set_aspect('equal')
ax1.set_title('Lag Plot (Lag_2) Energy Consumption', size = 18, weight = 'bold');
ax1.set_xlabel('Lag_2 (= 2 hours)', size = 16)
ax1.set_ylabel('Hourly Energy Consumption', size = 16)



#24
ax = sns.regplot(x='Lag_24', y='usage', data=df, ci=None, color = 'yellow', 
                 scatter_kws=dict(color='royalblue', alpha = 0.5), ax = ax3)
ax.set_aspect('equal')
ax.set_title('Lag Plot (Lag_24) Energy Consumption', size = 18, weight = 'bold');
ax.set_xlabel('Lag_24 (= one day)', size = 16)
ax.set_ylabel('Hourly Energy Consumption', size = 16)



#48
ax = sns.regplot(x='Lag_48', label = True,y='usage', data=df, ci=None, color = 'limegreen', 
                 scatter_kws=dict(color='royalblue', alpha = 0.5), ax = ax4)
ax.set_aspect('equal')

ax.set_title('Lag Plot (Lag_48) Energy Consumption', size = 18, weight = 'bold');
ax.set_xlabel('Lag_48 (= two days)', size = 16)
ax.set_ylabel('Hourly Energy Consumption', size = 16)
plt.legend(['Lag_48'])



#168
ax = sns.regplot(x='Lag_168', label = True,y='usage', data=df, ci=None, color = 'indigo', 
                 scatter_kws=dict(color='royalblue', alpha = 0.5), ax = ax5)
ax.set_aspect('equal')

ax.set_title('Lag Plot (Lag_168) Energy Consumption', size = 18, weight = 'bold');
ax.set_xlabel('Lag_168 (= one week)', size = 16)
ax.set_ylabel('Hourly Energy Consumption', size = 16)
plt.legend(['Lag_168'])



#672
ax = sns.regplot(x='Lag_672', label = True,y='usage', data=df, ci=None, color = 'violet', 
                 scatter_kws=dict(color='royalblue', alpha = 0.5), ax = ax6)
ax.set_aspect('equal')

ax.set_title('Lag Plot (Lag_672) Energy Consumption', size = 18, weight = 'bold');
ax.set_xlabel('Lag_672 (= one month)', size = 16)
ax.set_ylabel('Hourly Energy Consumption', size = 16)
plt.legend(['Lag_672'])

#1
plot(legend = True)
fig.savefig('Desktop/Thesis/export 3/usage/fotos/lagplots.png')   # save the figure to file


# In[ ]:





# In[ ]:



autocor1 = df['usage'].autocorr(lag = 1)
print(autocor1)

autocor2 = df['usage'].autocorr(lag = 2)
print(autocor2)

autocor12 = df['usage'].autocorr(lag = 12)
print(autocor12)

autocor24 = df['usage'].autocorr(lag = 24)
print(autocor24)

autocor48 = df['usage'].autocorr(lag = 48)
print(autocor48)

autocor168 = df['usage'].autocorr(lag = 168)
print(autocor168)

autocor672 = df['usage'].autocorr(lag = 672)
print(autocor672)


# In[ ]:


from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot


# In[ ]:


df.head()


# In[ ]:


dfcop = df[:300]


# In[ ]:


pyplot(df[:500]).autocorr()

pyplot.title('Autocorrelation Plot')
pyplot.show()


# In[14]:


print(f"Skewness: {df['usage'].skew()}")
print(f"Kurtosis: {df['usage'].kurt()}")


# From this information we see how the distribution:
# 
# does not follow a normal curve
# show spikes
# has kurtosis and asymmetry values greater than 1
# We do this for each variable, and we will have a pseudo-complete descriptive picture of their behavior.
# 
# 

# # Visualize data
# Date = df.index
# sns.set(rc={'figure.figsize':(10,7)})
# sns.lineplot(x=Date, y=df['usage'], color = 'royalblue', alpha = 0.9)
# plt.legend(['Usage per Hour'])
# plt.title(' Energy Comsumption Dataset (Time Series after cleaning)', size= 18, weight = 'bold')
# plt.xlabel('Hourly Date Range', size = 16)
# plt.ylabel(' Energy Consumption', size = 16)
# plt.savefig('Desktop/Thesis/export 3/usage/line.png')   # save the figure to file
# 

# ### Exploratory Data Analysis (EDA)

# #transforming DateTime column into index
# df2 = df.set_index('date')
# df2.index = pd.to_datetime(df2.index)

# Running the example plots the energy data (t) on the x-axis against the energy on the previous day (t-1) on the y-axis.

# In[23]:


values = DataFrame(df.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[24]:


values = DataFrame(df.values)
dataframe = concat([values.shift(2), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[25]:


values = DataFrame(df.values)
dataframe = concat([values.shift(24), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[26]:


values = DataFrame(df.values)
dataframe = concat([values.shift(48), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[27]:


values = DataFrame(df.values)
dataframe = concat([values.shift(168), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[28]:


lag_plot(df)
pyplot.show()


# In[ ]:


def create_features(df):
    """
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df
df = create_features(df)


# In[ ]:


df.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='usage')
ax.set_title('Energy usage by Hour')
plt.show()


# In[ ]:


dayofweek = 'Monday', 'Tuesday', 'Wednesday', 'Thurday', 'Friday', 'Saturday', 'Sunday'
_ = df.pivot_table(index=df['hour'], 
                     columns='dayofweek', 
                     values='usage',
                     aggfunc='sum').plot(figsize=(15,4),
                     title='Energy Consumption - Daily Trends')


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='dayofweek', y='usage', palette='Blues')
ax.set_title('Energy usage per weekday')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='usage', palette='Blues')
ax.set_title('MW by Month')
plt.show()


# _ = df2.pivot_table(index=df2['hour'],  
#                      values='usage',
#                      aggfunc='mean').plot(figsize=(6,4),
#                      title='Average Energy Consumption - Daily Trend',
#                                          xlabel = 'Hour of the day (00:00 till 23:00)',
#                                          ylabel = 'Mean - Energy Consumption')
# 
# 

# In[ ]:


title = 'hi'
_ = df.pivot_table(index=df['hour'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(8, 6),
                     title='Hourly Trend (per day)')
_.title.set_size(18)
_.title.set_weight('bold')
plt.xlabel('Hour of day (00:00 till 23:00)',size = 15)
plt.ylabel('Mean Usage Variable',size = 15)

plt.savefig('Desktop/Thesis/export 3/usage/AverageEnergyHOUR.png')   # save the figure to file



_ = df.pivot_table(index=df['dayofweek'],
                   values='usage',
                   aggfunc='mean').plot(figsize=(8,6),
                title='Daily Trend (per week)')
_.title.set_size(18)
_.title.set_weight('bold')
plt.xlabel('Weekdays (Sunday - Saturday)',size = 15)
plt.ylabel('Mean Usage Variable',size = 15)

plt.savefig('Desktop/Thesis/export 3/usage/AverageEnergyDAY.png')   # save the figure to file




_ = df.pivot_table(index=df['month'],
                values='usage',
                aggfunc='mean').plot(figsize=(8,6),
                title='Monthly Trend (per year)')
_.title.set_size(18)
_.title.set_weight('bold')
plt.xlabel('Month of the year (January - December)',size = 15)
plt.ylabel('Mean Usage Variable',size = 15)

plt.savefig('Desktop/Thesis/export 3/usage/AverageEnergyMONTH.png')   # save the figure to file




_ = df.pivot_table(index=df['weekofyear'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(8,6),
                     title='Weekly Year Trend (per year)')
_.title.set_size(18)
_.title.set_weight('bold')
plt.xlabel('Week of the Year (52)',size = 15)
plt.ylabel('Mean Usage Variable',size = 15)


plt.savefig('Desktop/Thesis/export 3/usage/AverageEnergyWEEK.png')   # save the figure to file



# _ = df.pivot_table(index=df['hour'],  
#                      values='usage',
#                      aggfunc='mean').plot(figsize=(6,4),
#                      title='Average Energy Consumption - Daily Trend')
# 
# _ = df.pivot_table(index=df['Dayofweek'],  
#                      values='usage',
#                      aggfunc='mean').plot(figsize=(6,4),
#                      title='Average Energy Consumption - Weekly Trend')
# 
# _ = df.pivot_table(index=df['month'],  
#                      values='usage',
#                      aggfunc='mean').plot(figsize=(6,4),
#                      title='Average Energy Consumption - Monthly Trend')
# 
# _ = df.pivot_table(index=df['weekofyear'],  
#                      values='usage',
#                      aggfunc='mean').plot(figsize=(6,4),
#                      title='Average Energy Consumption - Week of Year Trend')
# 
# 
# 

# In[ ]:


fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df2.loc[df2['quarter']==1].hour, df2.loc[df2['quarter']==1].usage)
ax.set_title('Hourly Boxplot PJME Q1')
ax.set_ylim(0,30)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df2.loc[df2['quarter']==2].hour, df2.loc[df2['quarter']==2].usage)
ax.set_title('Hourly Boxplot PJME Q2')
ax.set_ylim(0,30)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df2.loc[df2['quarter']==3].hour, df2.loc[df2['quarter']==3].usage)
ax.set_title('Hourly Boxplot PJME Q3')
ax.set_ylim(0,30)
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(df2.loc[df2['quarter']==4].hour, df2.loc[df2['quarter']==4].usage)
ax.set_title('Hourly Boxplot PJME Q4')
_ = ax.set_ylim(0,30)


# ### Rolling Mean

# In[ ]:


df2['usage24'] = df2['usage'].rolling(24).mean()
df2['usage168'] = df2['usage'].rolling(168).mean()
df2['usage672'] = df2['usage'].rolling(672).mean()


# In[ ]:


df2.head


# In[ ]:


df2.dtypes


# In[ ]:


# Visualize data usage
sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=df['hour'], y=df['usage'])


# In[ ]:


# Visualize data usage average
sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=df2['Dayofweek'], y=df2['usage24'])


# In[ ]:


# Visualize data usage average
sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=df2['Dayofweek'], y=df2['usage168'])


# In[ ]:


# Visualize data usage average
sns.set(rc={'figure.figsize':(12,8)})
sns.lineplot(x=df2['Dayofweek'], y=df2['usage'])
plt.legend(['usage'])


# In[ ]:


plt.subplot(1,3,1)
aux1 = df2[['hour', 'usage']].groupby( 'hour' ).sum().reset_index()
sns.barplot( x='hour', y='usage', data=aux1)

plt.subplot(1,3,2)
sns.heatmap( aux1.corr( method='pearson' ), annot=True );


# In[ ]:


pip install SCALECAST==0.1.5


# In[ ]:


pip install scalecast --upgrade


# In[ ]:


from scalecast.Forecaster import Forecaster


# In[ ]:


df.seasonal_decompose().plot()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Testing Stationarity
# Since the VAR model requires the time series you want to forecast to be stationary, it is customary to check all the time series in the system for stationarity.https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# Augmented Dickey-Fuller test An augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. It is basically Dickey-Fuller test with more lagged changes on RHS. https://analyticsindiamag.com/complete-guide-to-dickey-fuller-test-in-time-series-analysis/
# 

# In[31]:


from statsmodels.tsa.stattools import adfuller

parameters = ['adf', 'pvalue', 'used_lag', 'nobs']
adf_dict = {parameters[id]: val  for id, val in enumerate(adfuller(df)[:4])}
print(adf_dict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[23]:


series = df['usage'].values
series


# In[70]:


# ADF Test
result = adfuller(series, autolag='AIC')


print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")


# In[29]:


# Augmented Dickey-Fuller test 
adf = adfuller(df['usage'], autolag = 'AIC')
print("p-value of usage: {}".format(float(adf[1])))
print("LAG: {}".format(float(adf[2])))


# As it has p-value 2.8392813701514817e-07 which is less than 0.05, null hypothesis is rejected and this is not a random walk. Nog kijken: https://www.youtube.com/watch?v=_vQ0W_qXMxk

# In[95]:


fig, ax = plt.subplots(figsize=(10,7))

plot_acf(df['usage'], lags=50, ax=ax, color = 'royalblue')#plt. ipv plot
plt.ylim([0,1])
plt.yticks(np.arange(0.1, 1.1, 0.1))
plt.xticks(np.arange(0, 51, 2))
plt.title('Autocorrelation plot (ACF)', size = 18, weight = 'bold')
plt.xlabel('Number of lags', size = 16)
plt.ylabel('Correlation', size = 16)
plt.show()

fig.savefig('Desktop/Thesis/export 3/usage/fotos/acf.png')   # save the figure to file


# In[93]:


fig, ax = plt.subplots(figsize=(10,7))

plot_pacf(df['usage'], lags=50, ax=ax, color = 'royalblue')#plt. ipv plot

plt.yticks(np.arange(0.1, 1.1, 0.1))
plt.xticks(np.arange(0, 51, 2))
plt.title('Partial Autocorrelation plot (PACF)', size = 18, weight = 'bold')
plt.xlabel('Number of lags', size = 16)
plt.ylabel('Correlation', size = 16)
plt.show()

fig.savefig('Desktop/Thesis/export 3/usage/fotos/pacf.png')   # save the figure to file


# In[36]:


def tsplot1(y, title, lags= None, figsize=(12, 8)):
    fig = plt.figure(figsize = figsize)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout,(0, 0))
    hist_ax = plt.subplot2grid(layout,(0, 1))
    acf_ax = plt.subplot2grid(layout,(1, 0))
    pacf_ax = plt.subplot2grid(layout,(1, 1))
    
    y.plot(ax = ts_ax)
    ts_ax.set_title('hihi', fontsize = 18, fontweight = 'bold')
    y.plot(ax = hist_ax, kind = 'hist', bins = 25)
    hist_ax.set_title('Histogram')
    plot_acf(y, lags = lags, ax = acf_ax)#plt. ipv plot
    plot_pacf(y, lags = lags, ax = pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax


# In[37]:


tsplot1(df['usage'], title = 'hi', lags = 50)


# In[34]:


def tsplot2(y, title, lags= None, figsize=(12, 8)):
    fig = plt.figure(figsize = figsize)
    layout = (2,2)
    ts_ax = plt.subplot2grid(layout,(0, 0))
    hist_ax = plt.subplot2grid(layout,(0, 1))
    acf_ax = plt.subplot2grid(layout,(1, 0))
    pacf_ax = plt.subplot2grid(layout,(1, 1))
    
    y.plot(ax = ts_ax)
    ts_ax.set_title(title, fontsize = 14, fontweight = 'bold')
    y.plot(ax = hist_ax, kind = 'hist', bins = 25)
    hist_ax.set_title('Histogram')
    plot_acf(y, lags = lags, ax = acf_ax)#plt. ipv plot
    plot_pacf(y, lags = lags, ax = pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax


# In[35]:


tsplot2(df['usage'], title = 'hi', lags = 48)


# In[ ]:


num_var = len(df.iloc[1,:])
for i in range(0, num_var):
    tsplot2(df.iloc[:,i].dropna(), title = df.columns[i], lags = 50)


# In[ ]:


df.head()


# In[ ]:


dfnew = df.copy()
dfnew['usage'] = np.log(df.iloc[:,0]).diff(1)


# In[ ]:


pd.concat([df, dfnew], axis = 1).head(15)


# In[ ]:


for i in range(0, num_var):
    tsplot2(dfnew.iloc[:,i].dropna(), title = dfnew.columns[i], lags = 50)


# In[ ]:


train_size = int(len(df) * 0.80) 
test_size = len(df) - train_size
train, test = df[0:train_size,:], df[train_size:len(df),:]


# In[37]:


results = adfuller(df['usage'])
print(results)


# 0th element is test statistic(-1.34) = More negative means more likely to be stationary.
# 
# 1stelementisp-value:(0.60) = If p-value is small → reject null hypothesis. Reject non-stationary.
# 
# 4th element is the critical test statistics
# 

# In[67]:


dfnew = df.diff()
dfnew = df.diff().dropna()


# In[ ]:





# In[69]:


result = adfuller(df['usage'])
print(result)


# In[ ]:





# In[ ]:




