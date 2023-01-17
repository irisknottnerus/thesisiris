#!/usr/bin/env python
# coding: utf-8

# In[116]:


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


# advanced timeseries
# https://www.kaggle.com/code/davidanimaddo/energy-timeseries-advanced-data-visualization#1.-Introduction

# In[117]:


df = pd.read_csv('Desktop/Thesis/export 3/usage/elecweer.csv')


# In[118]:


#Formating to Date Time
df['date'] = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%Sz') for x in df['date']]


# In[119]:


#Check for missing values
print('Total num of missing values:') 
print(df.usage.isna().sum())
print('')
df_missing_date = df.loc[df.usage.isna() == True]


# df.index = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:&s')

# del df['date']

# In[130]:


df.index = pd.DatetimeIndex(df.index).to_period('H')


# In[ ]:


df.head()
df['X.6'].corr(df['usage'])


# In[120]:


df.head(24)
df = df.rename(columns={'X.2': 'DD', 'X.3': 'FH', 'X.4': 'FF', 'X.5': 'FX', 'X.6': 'temp', 'X.9': 'sun', 'X.14': 'sight', 'X.16': 'humi','X.10': 'q', 'X.11': 'DR', 'X.2': 'RH', 'X.18': 'IX','X.19': 'M', 'X.20': 'R','X.13': 'P','X.15': 'N' })


# In[108]:


del df['date']


# In[125]:


del df['X.21']


# In[126]:


del df['X.22']
del df['X.23']
del df['X.12']
del df['X.17']


# In[127]:


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


# In[131]:


df.head()


# In[123]:


# Selecting all rows till a new day starts at 00:00
df = df.iloc[13: , :]
df.head()
df.info()


# In[122]:


#transforming DateTime column into index
df = df.set_index('date')
df.index = pd.to_datetime(df.index)


# In[132]:


df.info()
df.describe().T


# # Initialize layout
# fig, ax = plt.subplots(figsize = (9, 7))
# median_hour = np.median(df['usage'])
# #plot
# ax.hist(df['usage'], bins=5, edgecolor="black", alpha = 0.8)
# #df2.index.hour
# ax.axvline(median_hour, color="black", ls="--", label="Median Usage")
# ax.set_title('Histogram - Distribution of variable: Usage', size = 18, weight = 'bold')
# ax.set_xlabel('Usage', size = 16)
# ax.set_ylabel('Density', size = 16)
# ax.legend(prop={'size': 12});
# 
# fig.savefig('Desktop/Thesis/export 3/usage/histo.png')   # save the figure to file
# plt.close(fig)  
# 
# 
# 
# sns_pp = sns.pairplot(df)
# sns_pp.savefig("sns-heatmap.png")
# 

# In[23]:


# Visualize data
Date = df['date']
sns.set(rc={'figure.figsize':(9,7)})
sns.lineplot(x=Date, y=df['X.6'], alpha = 0.9)
plt.legend(['usage'])
plt.title('Time Series after cleaning - Energy Comsumption Dataset', size= 18, weight = 'bold')
plt.xlabel('Date per hour', size = 16)
plt.ylabel(' Energy Consumption', size = 16)
plt.savefig('Desktop/Thesis/export 3/usage/line.png')   # save the figure to file



# In[48]:


# Visualize data
Date = df['date']
sns.set(rc={'figure.figsize':(9,7)})
sns.lineplot(x=Date, y=df['usage'], alpha = 0.9)
plt.legend(['usage'])
plt.title('Time Series after cleaning - Energy Comsumption Dataset', size= 18, weight = 'bold')
plt.xlabel('Date per hour', size = 16)
plt.ylabel(' Energy Consumption', size = 16)
plt.savefig('Desktop/Thesis/export 3/usage/line.png')   # save the figure to file


# ### Exploratory Data Analysis (EDA)

# In[24]:


#transforming DateTime column into index
df2 = df.set_index('date')
df2.index = pd.to_datetime(df2.index)


# In[9]:


#transforming DateTime column into index
df = df.set_index('date')
df.index = pd.to_datetime(df.index)


# Running the example plots the energy data (t) on the x-axis against the energy on the previous day (t-1) on the y-axis.

# In[ ]:


values = DataFrame(df.values)
dataframe = concat([values.shift(336), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[26]:


lag_plot(df)
pyplot.show()


# In[29]:


def create_features(df2):
    """
    """
    df2 = df2.copy()
    df2['hour'] = df2.index.hour
    df2['Dayofweek'] = df2.index.dayofweek
    df2['quarter'] = df2.index.quarter
    df2['month'] = df2.index.month
    df2['year'] = df2.index.year
    df2['dayofyear'] = df2.index.dayofyear
    df2['dayofmonth'] = df2.index.day
    df2['weekofyear'] = df2.index.isocalendar().week
    return df2
df2 = create_features(df2)


# In[27]:


def create_features(df):
    """
    """
    df1 = df.copy()
    df['hour'] = df.hour
    df2['Dayofweek'] = df.index.dayofweek
    return df1
df1 = create_features(df)


# In[ ]:


df2.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df2, x='hour', y='usage')
ax.set_title('Energy usage by Hour')
plt.show()


# In[ ]:


dayofweek = 'Monday', 'Tuesday', 'Wednesday', 'Thurday', 'Friday', 'Saturday', 'Sunday'
_ = df2.pivot_table(index=df2['hour'], 
                     columns='Dayofweek', 
                     values='usage',
                     aggfunc='sum').plot(figsize=(15,4),
                     title='Energy Consumption - Daily Trends')


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df2, x='Dayofweek', y='usage', palette='Blues')
ax.set_title('Energy usage per weekday')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df2, x='month', y='usage', palette='Blues')
ax.set_title('MW by Month')
plt.show()


# In[ ]:


_ = df2.pivot_table(index=df2['hour'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Daily Trend',
                                         xlabel = 'Hour of the day (00:00 till 23:00)',
                                         ylabel = 'Mean - Energy Consumption')


# In[ ]:


_ = df2.pivot_table(index=df2['hour'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Daily Trend',
                                         xlabel = 'Hour of the day (00:00 till 23:00)',
                                         ylabel = 'Mean - Energy Consumption')



_ = df2.pivot_table(index=df2['Dayofweek'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Weekly Trend',
                                         xlabel = 'Weekdays (sun - sat)')

_ = df2.pivot_table(index=df2['month'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Monthly Trend',
                                          xlabel = 'Month of the day (January - December)',
                                          ylabel = 'Mean - Energy Consumption')

_ = df2.pivot_table(index=df2['weekofyear'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Week of Year Trend',
                                          xlabel = 'Week of the year',
                                          ylabel = 'Mean - Energy Consumption')


plt.savefig('Desktop/Thesis/export 3/usage/line.png')   # save the figure to file



# In[ ]:





# In[ ]:


_ = df2.pivot_table(index=df2['hour'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Daily Trend')

_ = df2.pivot_table(index=df2['Dayofweek'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Weekly Trend')

_ = df2.pivot_table(index=df2['month'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Monthly Trend')

_ = df2.pivot_table(index=df2['weekofyear'],  
                     values='usage',
                     aggfunc='mean').plot(figsize=(6,4),
                     title='Average Energy Consumption - Week of Year Trend')



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
sns.lineplot(x=df2['hour'], y=df2['usage'])


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


# In[7]:


pip install SCALECAST==0.1.5


# In[25]:


pip install scalecast --upgrade


# In[ ]:


from scalecast.Forecaster import Forecaster


# In[ ]:


#using twi axes
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change["co2"])
ax.set_xlabel('Time')
ax.set_ylabel('CO2 (ppm)')
ax2 = ax.twinx()
ax2.plot(climate_change.index, climate_change["relative_temp"])
ax2.set_ylabel('Relative temperature (Celsius)')
plt.show()


# In[ ]:


#Separating variables by color
fig, ax = plt.subplots()
ax.plot(climate_change.index, climate_change["co2"], color='blue')
ax.set_xlabel('Time')ax.set_ylabel('CO2 (ppm)', color='blue')
ax2 = ax.twinx()
ax2.plot(climate_change.index, climate_change["relative_temp"],
         color='red')
ax2.set_ylabel('Relative temperature (Celsius)', color='red')plt.show()
#https://campus.datacamp.com/pdf/web/viewer.html?file=https://projector-video-pdf-converter.datacamp.com/13706/chapter2.pdf#page=21


# In[ ]:





# In[ ]:





# ### Testing Stationarity
# Since the VAR model requires the time series you want to forecast to be stationary, it is customary to check all the time series in the system for stationarity.https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# Augmented Dickey-Fuller test An augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. It is basically Dickey-Fuller test with more lagged changes on RHS. https://analyticsindiamag.com/complete-guide-to-dickey-fuller-test-in-time-series-analysis/
# 

# In[31]:


series = df['usage'].values
series


# In[32]:


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


# In[33]:


# Augmented Dickey-Fuller test 
adf = adfuller(df['usage'], autolag = 'AIC')
print("p-value of usage: {}".format(float(adf[1])))
print("LAG: {}".format(float(adf[2])))


# As it has p-value 2.8392813701514817e-07 which is less than 0.05, null hypothesis is rejected and this is not a random walk. Nog kijken: https://www.youtube.com/watch?v=_vQ0W_qXMxk

# In[34]:


from matplotlib import rcParams


# In[11]:


sns.pairplot(df)


# In[113]:


fig, ax = plt.subplots(figsize = (10,10))
corrmat = df.corr()
hm = sns.heatmap(corrmat, 
                 cbar=True, 
                 annot=True, 
                 square=True,
                 fmt='.2f', 
                 ax=ax,
                 annot_kws={'size': 9}, 
                 yticklabels=df.columns, 
                 xticklabels=df.columns, 
                 cmap="Blues")
plt.show()
plt.savefig('Desktop/Thesis/export 3/usage/fotos/corra.png') 


# #6 = T         = Temperatuur (in 0.1 graden Celsius) 
# #9 = SQ        = Duur van de zonneschijn (in 0.1 uren) per uurvak
# #10 = Q        = Globale straling (in J/cm2) per uurvak / Global radiation (in J/cm2) during the hourly division!
# #14 = VV       = Horizontaal zicht tijdens de waarneming (0=minder dan 100m

# In[ ]:


# Plotting a Correlation Plot for Data on Energy Consumption
corrplot(ecMatrix$r, type = "upper", order = "FPC", method = "color",
         p.mat = ecMatrix$P, sig.level = 0.01, insig = "pch",
         tl.cex = 0.8, tl.col = "black", tl.srt = 45)


# In the Correlation Plot shown above, the variables that are highly correlated are highlighted at the dark blue intersections. We used a level of significance of 0.01 to determine correlations that are statistically significant. Correlations with a p-value > 0.01 are considered statistically insignificant and marked with a cross.

# In[ ]:


# Normality Test
VAR_Est_norm <- normality.test(VAR_Est, multivariate.only = TRUE)
VAR_Est_norm
#https://rpubs.com/chenlianghe/607943

