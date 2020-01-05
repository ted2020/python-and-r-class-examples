import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt

tickers = ['MSFT', 'AAPL']
sec_data = pd.DataFrame()
for t in tickers:
    sec_data[t] = wb.DataReader(t, data_source='iex', start='2015-1-1')['close']

sec_data


# In[2]:


sec_data.info()


# Then, calculate the diversifiable and the non-diversifiable risk of a portfolio, composed of these two stocks:

# a) with weights 0.5 and 0.5;

# In[3]:


sec_returns = np.log(sec_data / sec_data.shift(1))
sec_returns


# ### Calculating Portfolio Variance

# Equal weightings scheme:

# In[4]:


weights = np.array([0.5, 0.5])


# Portfolio Variance:

# In[5]:


pfolio_var = np.dot(weights.T, np.dot(sec_returns.cov() * 250, weights))
pfolio_var


# ### Calculating Diversifiable and Non-Diversifiable Risk of a Portfolio

# Diversifiable Risk:

# In[6]:


MSFT_var_a = sec_returns[['MSFT']].var() * 250
MSFT_var_a


# In[7]:


AAPL_var_a = sec_returns[['AAPL']].var() * 250
AAPL_var_a


# Or:

# In[8]:


MSFT_var_a = sec_returns['MSFT'].var() * 250
MSFT_var_a


# In[9]:


AAPL_var_a = sec_returns['AAPL'].var() * 250
AAPL_var_a


# Calculating Diversifiable Risk:

# In[10]:


dr = pfolio_var - (weights[0] ** 2 * MSFT_var_a) - (weights[1] ** 2 * AAPL_var_a)
dr


# In[11]:


print (str(round(dr*100, 3)) + ' %')


# Calculating Non-Diversifiable Risk:

# In[12]:


n_dr_1 = pfolio_var - dr
n_dr_1


# Or:

# In[13]:


n_dr_2 = (weights[0] ** 2 * MSFT_var_a) + (weights[1] ** 2 * AAPL_var_a)
n_dr_2


# *****

# b)	With weights 0.2 for Microsoft and 0.8 for Apple.

# ### Calculating Portfolio Variance

# In[14]:


weights_2 = np.array([0.2, 0.8])


# Portfolio Variance:

# In[15]:


pfolio_var_2 = np.dot(weights_2.T, np.dot(sec_returns.cov() * 250, weights_2))
pfolio_var_2


# ### Calculating Diversifiable and Non-Diversifiable Risk of a Portfolio

# Calculating Diversifiable Risk:

# In[16]:


dr_2 = pfolio_var_2 - (weights_2[0] ** 2 * MSFT_var_a) - (weights_2[1] ** 2 * AAPL_var_a)
dr_2


# In[17]:


print (str(round(dr*100, 3)) + ' %')


# Calculating Non-Diversifiable Risk:

# In[18]:


n_dr_2 = pfolio_var_2 - dr_2
n_dr_2


# Or:

# In[19]:


n_dr_2 = (weights_2[0] ** 2 * MSFT_var_a) + (weights_2[1] ** 2 * AAPL_var_a)
n_dr_2
