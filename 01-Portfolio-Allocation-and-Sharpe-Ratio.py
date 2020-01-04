import pandas as pd


# In[2]:


import quandl


# ## Create a Portfolio

# In[3]:


start = pd.to_datetime('2012-01-01')
end = pd.to_datetime('2017-01-01')


# In[4]:


# Grabbing a bunch of tech stocks for our portfolio
aapl = quandl.get('WIKI/AAPL.11',
                  start_date = start,
                  end_date = end)
cisco = quandl.get('WIKI/CSCO.11',
                   start_date = start,
                   end_date = end)
ibm = quandl.get('WIKI/IBM.11',
                 start_date = start,
                 end_date = end)
amzn = quandl.get('WIKI/AMZN.11',
                  start_date = start,
                  end_date = end)


# In[5]:


# Alternative

aapl = pd.read_csv('AAPL_CLOSE',
                   index_col = 'Date',
                   parse_dates = True)
cisco = pd.read_csv('CISCO_CLOSE',
                    index_col = 'Date',
                    parse_dates = True)
ibm = pd.read_csv('IBM_CLOSE',
                  index_col = 'Date',
                  parse_dates = True)
amzn = pd.read_csv('AMZN_CLOSE',
                   index_col = 'Date',
                   parse_dates = True)


# In[6]:


'''
aapl.to_csv('AAPL_CLOSE')
cisco.to_csv('CISCO_CLOSE')
ibm.to_csv('IBM_CLOSE')
amzn.to_csv('AMZN_CLOSE')
'''


# ## Normalize Prices
#
# This is the same as cumulative daily returns

# In[7]:


# Example
aapl.iloc[0]['Adj. Close']


# In[8]:


for stock_df in (aapl, cisco, ibm, amzn):
    stock_df['Normed Return'] = stock_df['Adj. Close'] / stock_df.iloc[0]['Adj. Close']


# In[9]:


aapl.head()


# In[10]:


aapl.tail()


# ## Allocations
#
# Let's pretend we had the following allocations for our total portfolio:
#
# * 30% in Apple
# * 20% in Google/Alphabet
# * 40% in Amazon
# * 10% in IBM
#
# Let's have these values be reflected by multiplying our Norme Return by out Allocations

# In[11]:


for stock_df,allo in zip([aapl,cisco,ibm,amzn],[.3, .2, .4, .1]):
    stock_df['Allocation'] = stock_df['Normed Return'] * allo


# In[12]:


aapl.head()


# ## Investment
#
# Let's pretend we invested a million dollars in this portfolio

# In[13]:


for stock_df in [aapl,cisco,ibm,amzn]:
    stock_df['Position Values'] = stock_df['Allocation'] * 1000000


# ## Total Portfolio Value

# In[14]:


portfolio_val = pd.concat([aapl['Position Values'],
                           cisco['Position Values'],
                           ibm['Position Values'],
                           amzn['Position Values']],
                          axis = 1)


# In[15]:


portfolio_val.head()


# In[16]:


portfolio_val.columns = ['AAPL Pos', 'CISCO Pos', 'IBM Pos', 'AMZN Pos']


# In[17]:


portfolio_val.head()


# In[18]:


portfolio_val['Total Pos'] = portfolio_val.sum(axis = 1)


# In[19]:


portfolio_val.head()


# In[20]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


portfolio_val['Total Pos'].plot(figsize = (12, 8))
plt.title('Total Portfolio Value')


# In[22]:


portfolio_val.drop('Total Pos',
                   axis = 1).plot(kind = 'line', figsize = (12, 8))


# In[23]:


portfolio_val.tail()


# # Portfolio Statistics
# ### Daily Returns

# In[24]:


portfolio_val['Daily Return'] = portfolio_val['Total Pos'].pct_change(1)


# ### Cumulative Return

# In[25]:


cum_ret = 100 * (portfolio_val['Total Pos'][-1] / portfolio_val['Total Pos'][0] - 1 )
print('Our return {} was percent!'.format(cum_ret))


# ### Avg Daily Return

# In[26]:


portfolio_val['Daily Return'].mean()


# ### Std Daily Return

# In[27]:


portfolio_val['Daily Return'].std()


# In[28]:


portfolio_val['Daily Return'].plot(kind = 'kde')



SR = portfolio_val['Daily Return'].mean() / portfolio_val['Daily Return'].std()


# In[30]:


SR


# In[31]:


ASR = (252 ** 0.5) * SR


# In[32]:


ASR


# In[33]:


portfolio_val['Daily Return'].std()


# In[34]:


portfolio_val['Daily Return'].mean()


# In[35]:


fig = plt.figure(figsize = (12, 8))
portfolio_val['Daily Return'].plot('kde')


# In[36]:


fig = plt.figure(figsize = (12, 8))
aapl['Adj. Close'].pct_change(1).plot('kde', label = 'AAPL')
ibm['Adj. Close'].pct_change(1).plot('kde', label = 'IBM')
amzn['Adj. Close'].pct_change(1).plot('kde', label = 'AMZN')
cisco['Adj. Close'].pct_change(1).plot('kde', label = 'CISCO')
plt.legend()


# In[37]:


import numpy as np
np.sqrt(252) * (np.mean(.001 - 0.0002) / .001)
