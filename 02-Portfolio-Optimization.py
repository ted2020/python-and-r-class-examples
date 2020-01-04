import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Download and get Daily Returns
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


# In[3]:


stocks = pd.concat([aapl, cisco, ibm, amzn],
                   axis = 1)
stocks.columns = ['aapl','cisco','ibm','amzn']


# In[4]:


stocks.head()


# In[5]:


mean_daily_ret = stocks.pct_change(1).mean()
mean_daily_ret


# In[6]:


stocks.pct_change(1).corr()


# # Simulating Thousands of Possible Allocations

# In[7]:


stocks.head()


# In[8]:


stock_normed = stocks/stocks.iloc[0]
stock_normed.plot()


# In[9]:


stock_daily_ret = stocks.pct_change(1)
stock_daily_ret.head()


# ## Log Returns vs Arithmetic Returns
#
# We will now switch over to using log returns instead of arithmetic returns, for many of our use cases they are almost the same,but most technical analyses require detrending/normalizing the time series and using log returns is a nice way to do that.
# Log returns are convenient to work with in many of the algorithms we will encounter.
#
# For a full analysis of why we use log returns, check [this great article](https://quantivity.wordpress.com/2011/02/21/why-log-returns/).
#

# In[10]:


log_ret = np.log(stocks / stocks.shift(1))
log_ret.head()


# In[11]:


log_ret.hist(bins = 100,
             figsize = (12, 6));
plt.tight_layout()


# In[12]:


log_ret.describe().transpose()


# In[13]:


log_ret.mean() * 252


# In[14]:


# Compute pairwise covariance of columns
log_ret.cov()


# In[15]:


log_ret.cov() * 252 # multiply by days


# ## Single Run for Some Random Allocation

# In[16]:


# Set seed (optional)
np.random.seed(101)

# Stock Columns
print('Stocks')
print(stocks.columns)
print('\n')

# Create Random Weights
print('Creating Random Weights')
weights = np.array(np.random.random(4))
print(weights)
print('\n')

# Rebalance Weights
print('Rebalance to sum to 1.0')
weights = weights / np.sum(weights)
print(weights)
print('\n')

# Expected Return
print('Expected Portfolio Return')
exp_ret = np.sum(log_ret.mean() * weights) *252
print(exp_ret)
print('\n')

# Expected Variance
print('Expected Volatility')
exp_vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
print(exp_vol)
print('\n')

# Sharpe Ratio
SR = exp_ret/exp_vol
print('Sharpe Ratio')
print(SR)


# Great! Now we can just run this many times over!

# In[17]:


num_ports = 15000

all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):

    # Create Random Weights
    weights = np.array(np.random.random(4))

    # Rebalance Weights
    weights = weights / np.sum(weights)

    # Save Weights
    all_weights[ind,:] = weights

    # Expected Return
    ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

    # Expected Variance
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind] / vol_arr[ind]


# In[18]:


sharpe_arr.max()


# In[19]:


sharpe_arr.argmax()


# In[20]:


all_weights[1419,:]


# In[21]:


max_sr_ret = ret_arr[1419]
max_sr_vol = vol_arr[1419]


# ## Plotting the data

# In[22]:


plt.figure(figsize = (12, 8))
plt.scatter(vol_arr,
            ret_arr,
            c = sharpe_arr,
            cmap = 'plasma')
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Add red dot for max SR
plt.scatter(max_sr_vol,
            max_sr_ret,
            c = 'red',
            s = 50,
            edgecolors = 'black')


# # Mathematical Optimization
#
# There are much better ways to find good allocation weights than just guess and check! We can use optimization functions to find the ideal weights mathematically!

# ### Functionalize Return and SR operations

# In[23]:


def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])


# In[24]:


from scipy.optimize import minimize


# To fully understand all the parameters, check out:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

# In[25]:


help(minimize)


# Optimization works as a minimization function, since we actually want to maximize the Sharpe Ratio, we will need to turn it negative so we can minimize the negative sharpe (same as maximizing the postive sharpe)

# In[26]:


def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1


# In[27]:


# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1


# In[28]:


# By convention of minimize function it should be a function that returns zero for conditions
cons = ({'type' : 'eq', 'fun': check_sum})


# In[29]:


# 0-1 bounds for each weight
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))


# In[30]:


# Initial Guess (equal distribution)
init_guess = [0.25, 0.25, 0.25, 0.25]


# In[31]:


# Sequential Least SQuares Programming (SLSQP).
opt_results = minimize(neg_sharpe,
                       init_guess,
                       method = 'SLSQP',
                       bounds = bounds,
                       constraints = cons)


# In[32]:


opt_results


# In[33]:


opt_results.x


# In[34]:


get_ret_vol_sr(opt_results.x)


# # All Optimal Portfolios (Efficient Frontier)
#
# The efficient frontier is the set of optimal portfolios that offers the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. Portfolios that lie below the efficient frontier are sub-optimal, because they do not provide enough return for the level of risk. Portfolios that cluster to the right of the efficient frontier are also sub-optimal, because they have a higher level of risk for the defined rate of return.
#
# Efficient Frontier http://www.investopedia.com/terms/e/efficientfrontier

# In[35]:


# Our returns go from 0 to somewhere along 0.3
# Create a linspace number of points to calculate x on
frontier_y = np.linspace(0, 0.3, 100) # Change 100 to a lower number for slower computers!


# In[36]:


def minimize_volatility(weights):
    return  get_ret_vol_sr(weights)[1]


# In[37]:


frontier_volatility = []

for possible_return in frontier_y:
    # function for return
    cons = ({'type':'eq','fun': check_sum},
            {'type':'eq','fun': lambda w: get_ret_vol_sr(w)[0] - possible_return})

    result = minimize(minimize_volatility,
                      init_guess,
                      method = 'SLSQP',
                      bounds = bounds,
                      constraints = cons)

    frontier_volatility.append(result['fun'])


# In[38]:


plt.figure(figsize = (12, 8))
plt.scatter(vol_arr,
            ret_arr,
            c = sharpe_arr,
            cmap = 'plasma')
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')



# Add frontier line
plt.plot(frontier_volatility,
         frontier_y,
         'g--',
         linewidth = 3)
