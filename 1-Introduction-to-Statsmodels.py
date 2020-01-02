import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import statsmodels.api as sm


# In[3]:


df = sm.datasets.macrodata.load_pandas().data


# In[4]:


print(sm.datasets.macrodata.NOTE)


# In[5]:


df.head()


# In[6]:


index = pd.Index(sm.tsa.datetools.dates_from_range('1959Q1', '2009Q3'))


# In[7]:


df.index = index


# In[8]:


df.head()


# In[9]:


df['realgdp'].plot()
plt.ylabel("REAL GDP")


# Tuple unpacking
gdp_cycle, gdp_trend = sm.tsa.filters.hpfilter(df.realgdp)


# In[11]:


gdp_cycle


# In[12]:


type(gdp_cycle)


# In[13]:


df["trend"] = gdp_trend


# In[14]:


df[['trend','realgdp']].plot(figsize = (12, 8))


# In[15]:


df[['trend','realgdp']]["2000-03-31":].plot(figsize = (12, 8))


# ## Great job!
