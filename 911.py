
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data_911=pd.read_csv('911.csv')


# In[4]:


data_911.head()


# In[5]:


data_911.info()


# In[7]:


data_911['zip'].value_counts().head(5)


# In[8]:


data_911['twp'].value_counts().head(5)


# In[11]:


#len(data_911['title'].unique())
data_911['title'].nunique()


# In[18]:


dt1=data_911['title'][0]


# In[19]:


dt1


# In[21]:


dt1.split(':')[0]


# In[23]:


data_911['title'][0].split(':')[0]


# In[29]:


data_911['reason']=data_911['title'].apply(lambda x:x.split(':')[0])


# In[30]:


data_911['reason']


# In[31]:


data_911.head()


# In[32]:


data_911['reason'].value_counts()


# In[33]:


sns.countplot(x='reason',data=data_911)


# In[45]:


#data_911['timeStamp'].describe()
type(data_911['timeStamp'][0])
#type(data_911['timeStamp'].iloc[0])


# In[47]:


data_911['timeStamp_date']=pd.to_datetime(data_911['timeStamp'])


# In[48]:


data_911['timeStamp_date']


# In[51]:


date_time=data_911['timeStamp_date']


# In[52]:


date_time


# In[60]:


data_911['hour']=data_911['timeStamp_date'].apply(lambda x: x.hour)


# In[61]:


data_911['hour']


# In[62]:


data_911.head()


# In[63]:


datemap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[64]:


data_911['dayoftheweek']=data_911['timeStamp_date'].apply(lambda x: x.dayofweek)


# In[65]:


data_911['dayoftheweek']


# In[66]:


data_911.head()


# In[67]:


data_911['dayoftheweek']=data_911['dayoftheweek'].map(datemap)


# In[68]:


data_911.head()


# In[85]:


sns.countplot(x='dayoftheweek',data=data_911,hue='reason')
plt.tight_layout(pad=0.1)
plt.legend(borderaxespad=0)


# In[89]:


data_911['month']=data_911['timeStamp_date'].apply(lambda x: x.month)


# In[91]:


sns.countplot(x='month',data=data_911,hue='reason')


# In[95]:


bymonth=data_911.groupby('month').count()


# In[100]:


bymonth.head(12)


# In[101]:


bymonth['twp'].plot()


# In[107]:


sns.lmplot(x='month',y='twp',data=bymonth.reset_index())

