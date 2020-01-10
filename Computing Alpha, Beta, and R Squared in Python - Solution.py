import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_excel('D:/Python/Data_Files/IQ_data.xlsx')

X = data['Test 1']
Y = data['IQ']

plt.scatter(X,Y)
plt.axis([0, 120, 0, 150])
plt.ylabel('IQ')
plt.xlabel('Test 1')
plt.show()



# In[2]:


X1 = sm.add_constant(X)

reg = sm.OLS(Y, X1).fit()


# In[3]:


reg.summary()


# In[4]:


45 + 84*0.76

# In[5]:


slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)


# In[6]:


slope


# In[7]:


intercept


# In[8]:


r_value


# In[9]:


r_value ** 2


# In[10]:


p_value


# In[11]:


std_err


# In[12]:


intercept + 84 * slope

# In[13]:


def fitline(b):
    return intercept + slope * b


# In[14]:


line = fitline(X)

# In[15]:


plt.scatter(X,Y)
plt.plot(X,line)
plt.show()
