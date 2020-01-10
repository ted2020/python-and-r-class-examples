
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('monthly-milk-production-pounds-p.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# ** Clean Up**
#

# In[5]:


df.columns = ['Month', 'Milk in pounds per cow']
df.head()


# In[6]:


df.drop(168,
        axis = 0,
        inplace = True)


# In[7]:


df['Month'] = pd.to_datetime(df['Month'])


# In[8]:


df.head()


# In[9]:


df.set_index('Month',inplace=True)


# In[10]:


df.head()


# In[11]:


df.describe().transpose()


# ## Step 2: Visualize the Data
#
# In[12]:


df.plot()


# In[13]:


timeseries = df['Milk in pounds per cow']


# In[14]:


timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.rolling(12).std().plot(label='12 Month Rolling Std')
timeseries.plot()
plt.legend()


# In[15]:


timeseries.rolling(12).mean().plot(label = '12 Month Rolling Mean')
timeseries.plot()
plt.legend()


# ## Decomposition
#
# ETS decomposition allows us to see the individual parts!

# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['Milk in pounds per cow'],
                                   freq = 12)
fig = plt.figure()
fig = decomposition.plot()
fig.set_size_inches(15, 8)


# ## Testing for Stationarity
#
# In[17]:


df.head()


# In[18]:


from statsmodels.tsa.stattools import adfuller


# In[19]:


result = adfuller(df['Milk in pounds per cow'])


# In[20]:


print('Augmented Dickey-Fuller Test:')
labels = ['ADF Test Statistic',
          'p-value',
          '#Lags Used',
          'Number of Observations Used']

for value,label in zip(result,labels):
    print(label+' : '+str(value) )

if result[1] <= 0.05:
    print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
else:
    print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[21]:


def adf_check(time_series):
    """
    Pass in a time series, returns ADF report
    """
    result = adfuller(time_series)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic',
              'p-value',
              '#Lags Used',
              'Number of Observations Used']

    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")


# In[22]:


df['Milk First Difference'] = df['Milk in pounds per cow'] - df['Milk in pounds per cow'].shift(1)


# In[23]:


adf_check(df['Milk First Difference'].dropna())


# In[24]:


df['Milk First Difference'].plot()



# In[25]:

df['Milk Second Difference'] = df['Milk First Difference'] - df['Milk First Difference'].shift(1)


# In[26]:


adf_check(df['Milk Second Difference'].dropna())


# In[27]:


df['Milk Second Difference'].plot()


# ** Seasonal Difference **

# In[28]:


df['Seasonal Difference'] = df['Milk in pounds per cow'] - df['Milk in pounds per cow'].shift(12)
df['Seasonal Difference'].plot()


# In[29]:


# Seasonal Difference by itself was not enough!
adf_check(df['Seasonal Difference'].dropna())


# ** Seasonal First Difference **

# In[30]:


# You can also do seasonal first difference
df['Seasonal First Difference'] = df['Milk First Difference'] - df['Milk First Difference'].shift(12)
df['Seasonal First Difference'].plot()


# In[31]:


adf_check(df['Seasonal First Difference'].dropna())

# In[32]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf


# In[33]:

fig_first = plot_acf(df["Milk First Difference"].dropna())


# In[34]:


fig_seasonal_first = plot_acf(df["Seasonal First Difference"].dropna())


# In[35]:


from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df['Seasonal First Difference'].dropna())


# In[36]:


result = plot_pacf(df["Seasonal First Difference"].dropna())

# In[37]:


fig = plt.figure(figsize = (12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal First Difference'].iloc[13:],
                               lags = 40,
                               ax = ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal First Difference'].iloc[13:],
                                lags = 40,
                                ax = ax2)

# In[38]:


# For non-seasonal data
from statsmodels.tsa.arima_model import ARIMA


# In[39]:



help(ARIMA)

# In[40]:


# We have seasonal data!
model = sm.tsa.statespace.SARIMAX(df['Milk in pounds per cow'],
                                  order = (0,1,0),
                                  seasonal_order = (1,1,1,12))
results = model.fit()
print(results.summary())


# In[41]:


results.resid.plot()


# In[42]:


results.resid.plot(kind = 'kde')

# In[43]:


df['forecast'] = results.predict(start = 150,
                                 end = 168,
                                 dynamic = True)
df[['Milk in pounds per cow','forecast']].plot(figsize = (12, 8))

# In[44]:


df.tail()


# In[45]:

# In[46]:


from pandas.tseries.offsets import DateOffset


# In[47]:


future_dates = [df.index[-1] + DateOffset(months = x) for x in range(0,24) ]


# In[48]:


future_dates


# In[49]:


future_dates_df = pd.DataFrame(index = future_dates[1:],
                               columns = df.columns)


# In[50]:


future_df = pd.concat([df,future_dates_df])


# In[51]:


future_df.head()


# In[52]:


future_df.tail()


# In[53]:


future_df['forecast'] = results.predict(start = 168,
                                        end = 188,
                                        dynamic= True)
future_df[['Milk in pounds per cow', 'forecast']].plot(figsize = (12, 8))
