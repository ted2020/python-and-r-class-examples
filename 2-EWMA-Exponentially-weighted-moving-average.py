import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


airline = pd.read_csv('airline_passengers.csv',
                      index_col = "Month")


# In[3]:


airline.dropna(inplace = True)
airline.index = pd.to_datetime(airline.index)


# In[4]:


airline.head()


# # SMA
# ## Simple Moving Average

airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window = 6).mean()
airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window = 12).mean()


# In[6]:


airline.head()


# In[7]:


airline.plot(figsize = (12, 8))

# In[8]:


airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span = 12).mean()


# In[9]:


airline[['Thousands of Passengers','EWMA12']].plot(figsize = (12, 8))
