
from pylab import plt
plt.style.use('ggplot')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
# import warnings; warnings.simplefilter('ignore')


# ## Normality Tests

# ### Benchmark Case

# In[2]:


import numpy as np
np.random.seed(1000)
import scipy.stats as scs
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def gen_paths(S0, r, sigma, T, M, I):
    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        rand = (rand - rand.mean()) / rand.std()
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt +
                                         sigma * np.sqrt(dt) * rand)
    return paths


# In[4]:


S0 = 100.
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 250000


# In[5]:


paths = gen_paths(S0, r, sigma, T, M, I)


# In[6]:


plt.plot(paths[:, :10])
plt.grid(True)
plt.xlabel('time steps')
plt.ylabel('index level')
# tag: normal_sim_1
# title: 10 simulated paths of geometric Brownian motion


# In[7]:


log_returns = np.log(paths[1:] / paths[0:-1])


# In[8]:


paths[:, 0].round(4)


# In[9]:


log_returns[:, 0].round(4)


# In[10]:


def print_statistics(array):
    ''' Prints selected statistics.

    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    sta = scs.describe(array)
    print("%14s %15s" % ('statistic', 'value'))
    print(30 * "-")
    print("%14s %15.5f" % ('size', sta[0]))
    print("%14s %15.5f" % ('min', sta[1][0]))
    print("%14s %15.5f" % ('max', sta[1][1]))
    print("%14s %15.5f" % ('mean', sta[2]))
    print("%14s %15.5f" % ('std', np.sqrt(sta[3])))
    print("%14s %15.5f" % ('skew', sta[4]))
    print("%14s %15.5f" % ('kurtosis', sta[5]))


# In[11]:


print_statistics(log_returns.flatten())


# In[12]:


plt.hist(log_returns.flatten(), bins=70, normed=True, label='frequency')
plt.grid(True)
plt.xlabel('log-return')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, loc=r / M, scale=sigma / np.sqrt(M)),
         'r', lw=2.0, label='pdf')
plt.legend()
# tag: normal_sim_2
# title: Histogram of log-returns and normal density function


# In[13]:


sm.qqplot(log_returns.flatten()[::500], line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
# tag: sim_val_qq_1
# title: Quantile-quantile plot for log returns


# In[14]:


def normality_tests(arr):
    ''' Tests for normality distribution of given data set.

    Parameters
    ==========
    array: ndarray
        object to generate statistics on
    '''
    print("Skew of data set  %14.3f" % scs.skew(arr))
    print("Skew test p-value %14.3f" % scs.skewtest(arr)[1])
    print("Kurt of data set  %14.3f" % scs.kurtosis(arr))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(arr)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(arr)[1])


# In[15]:


normality_tests(log_returns.flatten())


# In[16]:


f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.hist(paths[-1], bins=30)
ax1.grid(True)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.set_title('regular data')
ax2.hist(np.log(paths[-1]), bins=30)
ax2.grid(True)
ax2.set_xlabel('log index level')
ax2.set_title('log data')
# tag: normal_sim_3
# title: Histogram of simulated end-of-period index levels
# size: 90


# In[17]:


print_statistics(paths[-1])


# In[18]:


print_statistics(np.log(paths[-1]))


# In[19]:


normality_tests(np.log(paths[-1]))


# In[20]:


log_data = np.log(paths[-1])
plt.hist(log_data, bins=70, normed=True, label='observed')
plt.grid(True)
plt.xlabel('index levels')
plt.ylabel('frequency')
x = np.linspace(plt.axis()[0], plt.axis()[1])
plt.plot(x, scs.norm.pdf(x, log_data.mean(), log_data.std()),
         'r', lw=2.0, label='pdf')
plt.legend()
# tag: normal_sim_4
# title: Histogram of log index levels and normal density function


# In[21]:


sm.qqplot(log_data, line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
# tag: sim_val_qq_2
# title: Quantile-quantile plot for log index levels


# ### Real World Data

# In[22]:


import pandas as pd


# In[23]:


raw = pd.read_csv('source/tr_eikon_eod_data.csv',
                 index_col=0, parse_dates=True)


# In[24]:


symbols = ['SPY', 'GLD', 'AAPL.O', 'MSFT.O']


# In[25]:


data = raw[symbols]
data = data.dropna()


# In[26]:


data.info()


# In[27]:


data.head()


# In[28]:


(data / data.ix[0] * 100).plot(figsize=(8, 6), grid=True)
# tag: real_returns_1
# title: Evolution of stock and index levels over time


# In[29]:


log_returns = np.log(data / data.shift(1))
log_returns.head()


# In[30]:


log_returns.hist(bins=50, figsize=(9, 6))
# tag: real_returns_2
# title: Histogram of respective log-returns
# size: 90


# In[31]:


for sym in symbols:
    print("\nResults for symbol %s" % sym)
    print(30 * "-")
    log_data = np.array(log_returns[sym].dropna())
    print_statistics(log_data)


# In[32]:


sm.qqplot(log_returns['SPY'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
# tag: real_val_qq_1
# title: Quantile-quantile plot for S&P 500 log returns


# In[33]:


sm.qqplot(log_returns['MSFT.O'].dropna(), line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')
# tag: real_val_qq_2
# title: Quantile-quantile plot for Microsoft log returns


# In[34]:


for sym in symbols:
    print("\nResults for symbol %s" % sym)
    print(32 * "-")
    log_data = np.array(log_returns[sym].dropna())
    normality_tests(log_data)


# ## Portfolio Optimization

# ### The Data

# In[35]:


symbols = ['AAPL.O', 'MSFT.O', 'AMZN.O', 'GDX', 'GLD']
noa = len(symbols)


# In[36]:


data = raw[symbols]


# In[37]:


(data / data.ix[0] * 100).plot(figsize=(8, 5), grid=True)
# tag: portfolio_1
# title: Stock prices over time
# size: 90


# In[38]:


rets = np.log(data / data.shift(1))


# In[39]:


rets.mean() * 252


# In[40]:


rets.cov() * 252


# ### The Basic Theory

# In[41]:


weights = np.random.random(noa)
weights /= np.sum(weights)


# In[42]:


weights


# In[43]:


np.sum(rets.mean() * weights) * 252
  # expected portfolio return


# In[44]:


np.dot(weights.T, np.dot(rets.cov() * 252, weights))
  # expected portfolio variance


# In[45]:


np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
  # expected portfolio standard deviation/volatility


# In[46]:


prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T,
                        np.dot(rets.cov() * 252, weights))))
prets = np.array(prets)
pvols = np.array(pvols)


# In[47]:


plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
# tag: portfolio_2
# title: Expected return and volatility for different/random portfolio weights
# size: 90


# ### Portfolio Optimizations

# In[48]:


def statistics(weights):
    ''' Return portfolio statistics.

    Parameters
    ==========
    weights : array-like
        weights for different securities in portfolio

    Returns
    =======
    pret : float
        expected portfolio return
    pvol : float
        expected portfolio volatility
    pret / pvol : float
        Sharpe ratio for rf=0
    '''
    weights = np.array(weights)
    pret = np.sum(rets.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])


# In[49]:


import scipy.optimize as sco


# In[50]:


def min_func_sharpe(weights):
    return -statistics(weights)[2]


# In[51]:


cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})


# In[52]:


bnds = tuple((0, 1) for x in range(noa))


# In[53]:


noa * [1. / noa,]


# In[54]:


get_ipython().run_cell_magic('time', '', "opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',\n                       bounds=bnds, constraints=cons)")


# In[55]:


opts


# In[56]:


opts['x'].round(3)


# In[57]:


statistics(opts['x']).round(3)


# In[58]:


def min_func_variance(weights):
    return statistics(weights)[1] ** 2


# In[59]:


optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP',
                       bounds=bnds, constraints=cons)


# In[60]:


optv


# In[61]:


optv['x'].round(3)


# In[62]:


statistics(optv['x']).round(3)


# ### Efficient Frontier

# In[63]:


cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in weights)


# In[64]:


def min_func_port(weights):
    return statistics(weights)[1]


# In[65]:


get_ipython().run_cell_magic('time', '', "trets = np.linspace(0.0, 0.25, 50)\ntvols = []\nfor tret in trets:\n    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},\n            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})\n    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',\n                       bounds=bnds, constraints=cons)\n    tvols.append(res['fun'])\ntvols = np.array(tvols)")


# In[66]:


plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets,
            c=prets / pvols, marker='o')
            # random portfolio composition
plt.scatter(tvols, trets,
            c=trets / tvols, marker='x')
            # efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
            # portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
            # minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
# tag: portfolio_3
# title: Minimum risk portfolios for given return level (crosses)
# size: 90


# ### Capital Market Line

# In[67]:


import scipy.interpolate as sci


# In[68]:


ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]


# In[69]:


tck = sci.splrep(evols, erets)


# In[70]:


def f(x):
    ''' Efficient frontier function (splines approximation). '''
    return sci.splev(x, tck, der=0)
def df(x):
    ''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der=1)


# In[71]:


def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3


# In[72]:


opt = sco.fsolve(equations, [0.01, 0.5, 0.15])


# In[73]:


opt


# In[74]:


np.round(equations(opt), 6)


# In[75]:


plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets,
            c=(prets - 0.01) / pvols, marker='o')
            # random portfolio composition
plt.plot(evols, erets, 'g', lw=4.0)
            # efficient frontier
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
            # capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
# tag: portfolio_4
# title: Capital market line and tangency portfolio (star) for risk-free rate of 1%
# size: 90


# In[76]:


cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                       bounds=bnds, constraints=cons)


# In[77]:


res['x'].round(3)
