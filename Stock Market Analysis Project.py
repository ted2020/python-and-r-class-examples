import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ____
# ## Part 1: Getting the Data
#
# ### Tesla Stock (Ticker: TSLA on the NASDAQ)
#

import pandas_datareader.data as web
import datetime
start = datetime.datetime(2012, 1, 1)
end = datetime.datetime(2017, 1, 1)
tesla_stock = web.DataReader('TSLA', 'google', start, end)
tesla_stock.head()


# In[3]:


# CSV files is used since Google Finance does not provide full data.
tesla_stock = pd.read_csv('./Tesla_Stock.csv',
                          index_col= 'Date')


# In[4]:


tesla_stock.head()


# ### Other Car Companies
#
# ** Repeat the same steps to grab data for Ford and GM (General Motors), **

# In[5]:


ford_stock = pd.read_csv('./Ford_Stock.csv', index_col= 'Date')


# In[6]:


ford_stock.head()


# In[7]:


gm_stock = pd.read_csv('./GM_Stock.csv', index_col= 'Date')


# In[8]:


gm_stock.head()


# ## Part 2: Visualizing the Data

# In[9]:


# Code Here
fig = plt.figure(figsize = (12, 6))
plt.title('Open')

tesla_stock['Open'].plot(label = 'Tesla')
ford_stock['Open'].plot(label = 'Ford')
gm_stock['Open'].plot(label = 'GM')
plt.legend()


# ____

# ** Plot the Volume of stock traded each day.**

# In[10]:


fig = plt.figure(figsize = (12, 6))
plt.title('Volume')

tesla_stock['Volume'].plot(label = 'Tesla')
ford_stock['Volume'].plot(label = 'Ford')
gm_stock['Volume'].plot(label = 'GM')
plt.legend()


# In[11]:


ford_stock['Volume'].argmax()


# In[12]:


# In[13]:


# Code Here
tesla_stock['Total Traded'] = tesla_stock['Open'] * tesla_stock['Volume']
ford_stock['Total Traded'] = ford_stock['Open'] * ford_stock['Volume']
gm_stock['Total Traded'] = gm_stock['Open'] * gm_stock['Volume']


# In[14]:


# Code here
fig = plt.figure(figsize = (12, 6))
plt.title('Total Traded')

tesla_stock['Total Traded'].plot(label = 'Tesla')
ford_stock['Total Traded'].plot(label = 'Ford')
gm_stock['Total Traded'].plot(label = 'GM')
plt.legend()


# In[ ]:




# In[15]:


tesla_stock['Total Traded'].argmax()


# In[16]:


# In[17]:


# In[18]:


fig = plt.figure(figsize = (12, 6))
gm_stock.rolling(window = 50).mean()['Open'].plot(label = 'MA50')
gm_stock.rolling(window = 200).mean()['Open'].plot(label = 'MA200')
plt.legend()

# In[19]:


from pandas.plotting import scatter_matrix


# In[20]:


df = pd.concat([tesla_stock['Open'], ford_stock['Open'], gm_stock['Open']], axis = 1)
df.columns = ['Tesla', 'Ford', 'GM']
df.head()


# In[21]:


scatter_matrix(df, figsize = (10, 10), hist_kwds = {'bins' : 100})


# In[22]:


start = '2012-01'
end = '2012-02'
ford_candle = ford_stock.loc[start:end]


# In[23]:

# In[24]:


tesla_stock['returns'] = (tesla_stock['Close'] / tesla_stock['Close'].shift(1)) - 1


# In[25]:


tesla_stock.head()


# In[26]:


ford_stock['returns'] = (ford_stock['Close'] / ford_stock['Close'].shift(1)) - 1
gm_stock['returns'] = (gm_stock['Close'] / gm_stock['Close'].shift(1)) - 1


# In[27]:


# Separately
fig = plt.figure(0)
tesla_stock['returns'].plot(kind = 'hist', bins = 50)
plt.title('Tesla')

plt.show()

fig = plt.figure(1)
ford_stock['returns'].plot(kind = 'hist', bins = 50)
plt.title('Ford')
plt.show()

fig = plt.figure(2)
gm_stock['returns'].plot(kind = 'hist', bins = 50)
plt.title('GM')
plt.show()


# In[28]:


# On one graph.
fig = plt.figure(figsize = (12, 10))
tesla_stock['returns'].plot(kind = 'hist',
                            bins = 50,
                            label = 'Tesla',
                            alpha = 0.5)
ford_stock['returns'].plot(kind = 'hist',
                            bins = 50,
                            label = 'Ford',
                            alpha = 0.8)
gm_stock['returns'].plot(kind = 'hist',
                            bins = 50,
                            label = 'GM',
                            alpha = 0.4)
plt.legend()



# In[29]:


fig = plt.figure(figsize = (12, 10))
tesla_stock['returns'].plot(kind = 'kde',
                            label = 'Tesla',
                            alpha = 0.5)
ford_stock['returns'].plot(kind = 'kde',
                           label = 'Ford',
                           alpha = 0.8)
gm_stock['returns'].plot(kind = 'kde',
                         label = 'GM',
                         alpha = 0.4)
plt.legend()



# In[30]:


box_df = pd.concat([tesla_stock['returns'], ford_stock['returns'], gm_stock['returns']], axis = 1)
box_df.columns = ['Tesla', 'Ford', 'GM']
box_df.plot(kind = 'box', figsize = (12, 10))
plt.legend()


# ## Comparing Daily Returns between Stocks
#
# In[31]:


from pandas.plotting import scatter_matrix

scatter_matrix(box_df, figsize = (10, 10), hist_kwds={'bins':50})


# ** It looks like Ford and GM do have some sort of possible relationship, let's plot just these two against eachother in scatter plot to view this more closely!**

# In[32]:


fig = plt.figure(figsize = (12, 8))
plt.scatter(ford_stock['returns'], gm_stock['returns'])

# In[33]:


tesla_stock['Cumulative Return'] = (1 + tesla_stock['returns']).cumprod()
tesla_stock.head()


# In[34]:


ford_stock['Cumulative Return'] = (1 + ford_stock['returns']).cumprod()
gm_stock['Cumulative Return'] = (1 + gm_stock['returns']).cumprod()



# In[35]:


fig = plt.figure(figsize = (12, 6))
tesla_stock['Cumulative Return'].plot(label = 'Tesla')
ford_stock['Cumulative Return'].plot(label = 'Ford')
gm_stock['Cumulative Return'].plot(label = 'GM')
plt.legend()
