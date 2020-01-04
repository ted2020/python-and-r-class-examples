
from scipy import stats


# In[3]:


help(stats.linregress)


# In[4]:


import pandas as pd


# In[5]:


import pandas_datareader as web


# In[6]:


spy_etf = web.DataReader('SPY', 'google')


# In[7]:


spy_etf.info()


# In[8]:


spy_etf.head()


# In[9]:


start = pd.to_datetime('2010-01-04')
end = pd.to_datetime('2017-07-18')


# In[10]:


aapl = web.DataReader('AAPL', 'google', start, end)


# In[11]:


aapl.head()


# In[12]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


aapl['Close'].plot(label = 'AAPL',
                   figsize = (12, 8))
spy_etf['Close'].plot(label = 'SPY Index')
plt.legend()


# ## Compare Cumulative Return

# In[14]:


aapl['Cumulative'] = aapl['Close'] / aapl['Close'].iloc[0]
spy_etf['Cumulative'] = spy_etf['Close'] / spy_etf['Close'].iloc[0]


# In[15]:


aapl['Cumulative'].plot(label = 'AAPL',
                        figsize = (10,8))
spy_etf['Cumulative'].plot(label = 'SPY Index')
plt.legend()
plt.title('Cumulative Return')


# ## Get Daily Return

# In[16]:


aapl['Daily Return'] = aapl['Close'].pct_change(1)
spy_etf['Daily Return'] = spy_etf['Close'].pct_change(1)


# In[17]:


fig = plt.figure(figsize = (12, 8))
plt.scatter(aapl['Daily Return'], spy_etf['Daily Return'],
            alpha = 0.3)


# In[18]:


aapl['Daily Return'].hist(bins = 100, figsize = (12, 8))


# In[19]:


spy_etf['Daily Return'].hist(bins = 100, figsize = (12, 8))


# In[20]:


beta,alpha, r_value, p_value, std_err = stats.linregress(aapl['Daily Return'].iloc[1:],spy_etf['Daily Return'].iloc[1:])


# In[21]:


beta


# In[22]:


alpha


# In[23]:


r_value


# ## What if our stock was completely related to SP500?

# In[24]:


spy_etf['Daily Return'].head()


# In[25]:


import numpy as np


# In[26]:


noise = np.random.normal(0, 0.001, len(spy_etf['Daily Return'].iloc[1:]))


# In[27]:


noise


# In[28]:


spy_etf['Daily Return'].iloc[1:] + noise


# In[29]:


beta, alpha, r_value, p_value, std_err = stats.linregress(spy_etf['Daily Return'].iloc[1:]+noise,
                                                      spy_etf['Daily Return'].iloc[1:])


# In[30]:


beta


# In[31]:


alpha
