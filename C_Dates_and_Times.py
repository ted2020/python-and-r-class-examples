
from pylab import plt
plt.style.use('seaborn')
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'


# ## Python

# In[2]:


import datetime as dt


# In[3]:


dt.datetime.now()


# In[4]:


to = dt.datetime.today()
to


# In[5]:


type(to)


# In[6]:


dt.datetime.today().weekday()
  # zero numbering 0 = Monday


# In[7]:


d = dt.datetime(2016, 10, 31, 10, 5, 30, 500000)
d


# In[8]:


print(d)


# In[9]:


str(d)


# In[10]:


d.year


# In[11]:


d.month


# In[12]:


d.day


# In[13]:


d.hour


# In[14]:


o = d.toordinal()
o


# In[15]:


dt.datetime.fromordinal(o)


# In[16]:


t = dt.datetime.time(d)
t


# In[17]:


type(t)


# In[18]:


dd = dt.datetime.date(d)
dd


# In[19]:


d.replace(second=0, microsecond=0)


# In[20]:


td = d - dt.datetime.now()
td


# In[21]:


type(td)


# In[22]:


td.days


# In[23]:


td.seconds


# In[24]:


td.microseconds


# In[25]:


td.total_seconds()


# In[26]:


d.isoformat()


# In[27]:


d.strftime("%A, %d. %B %Y %I:%M%p")


# In[28]:


dt.datetime.strptime('2017-03-31', '%Y-%m-%d')
  # year first and four digit year


# In[29]:


dt.datetime.strptime('30-4-16', '%d-%m-%y')
  # day first and two digit year


# In[30]:


ds = str(d)
ds


# In[31]:


dt.datetime.strptime(ds, '%Y-%m-%d %H:%M:%S.%f')


# In[32]:


dt.datetime.now()


# In[33]:


dt.datetime.utcnow()
  #  Universal Time, Coordinated


# In[34]:


dt.datetime.now() - dt.datetime.utcnow()
  # UTC + 2h = CET (summer)


# In[35]:


class UTC(dt.tzinfo):
    def utcoffset(self, d):
        return dt.timedelta(hours=0)
    def dst(self, d):
        return dt.timedelta(hours=0)
    def tzname(self, d):
        return "UTC"


# In[36]:


u = dt.datetime.utcnow()
u = u.replace(tzinfo=UTC())
  # attach time zone information
u


# In[37]:


class CET(dt.tzinfo):
    def utcoffset(self, d):
        return dt.timedelta(hours=2)
    def dst(self, d):
        return dt.timedelta(hours=1)
    def tzname(self, d):
        return "CET + 1"


# In[38]:


u.astimezone(CET())


# In[39]:


import pytz


# In[40]:


pytz.country_names['US']


# In[41]:


pytz.country_timezones['BE']


# In[42]:


pytz.common_timezones[-10:]


# In[43]:


u = dt.datetime.utcnow()
u = u.replace(tzinfo=pytz.utc)
u


# In[44]:


u.astimezone(pytz.timezone("CET"))


# In[45]:


u.astimezone(pytz.timezone("GMT"))


# In[46]:


u.astimezone(pytz.timezone("US/Central"))


# ## NumPy

# In[47]:


import numpy as np


# In[48]:


nd = np.datetime64('2015-10-31')
nd


# In[49]:


np.datetime_as_string(nd)


# In[50]:


np.datetime_data(nd)


# In[51]:


d


# In[52]:


nd = np.datetime64(d)
nd


# In[53]:


nd.astype(dt.datetime)


# In[54]:


nd = np.datetime64('2015-10', 'D')
nd


# In[55]:


np.datetime64('2015-10') == np.datetime64('2015-10-01')


# In[56]:


np.array(['2016-06-10', '2016-07-10', '2016-08-10'], dtype='datetime64')


# In[57]:


np.array(['2016-06-10T12:00:00', '2016-07-10T12:00:00',
          '2016-08-10T12:00:00'], dtype='datetime64[s]')


# In[58]:


np.arange('2016-01-01', '2016-01-04', dtype='datetime64')
  # daily frequency as default in this case


# In[59]:


np.arange('2016-01-01', '2016-10-01', dtype='datetime64[M]')
  # monthly frequency


# In[60]:


np.arange('2016-01-01', '2016-10-01', dtype='datetime64[W]')[:10]
  # weekly frequency


# In[61]:


dtl = np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
                dtype='datetime64[h]')
  # hourly frequency
dtl[:10]


# In[62]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


np.random.seed(3000)
rnd = np.random.standard_normal(len(dtl)).cumsum() ** 2


# In[64]:


fig = plt.figure(figsize=(10, 6))
plt.plot(dtl.astype(dt.datetime), rnd)
  # convert np.datetime to datetime.datetime
plt.grid(True)
fig.autofmt_xdate();
  # auto formatting of datetime xticks
# tag: datetime_plot
# title: Plot with datetime.datetime xticks auto-formatted


# In[65]:


np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
          dtype='datetime64[s]')[:10]
  # seconds as frequency


# In[66]:


np.arange('2016-01-01T00:00:00', '2016-01-02T00:00:00',
          dtype='datetime64[ms]')[:10]
  # milliseconds as frequency


# ## pandas

# In[67]:


import pandas as pd


# In[68]:


ts = pd.Timestamp('2016-06-30')
ts


# In[69]:


d = ts.to_pydatetime()
d


# In[70]:


pd.Timestamp(d)


# In[71]:


pd.Timestamp(nd)


# In[72]:


dti = pd.date_range('2016/01/01', freq='M', periods=12)
dti


# In[73]:


dti[6]


# In[74]:


pdi = dti.to_pydatetime()
pdi


# In[75]:


pd.DatetimeIndex(pdi)


# In[76]:


pd.DatetimeIndex(dtl.astype(pd.datetime))


# In[77]:


rnd = np.random.standard_normal(len(dti)).cumsum() ** 2


# In[78]:


df = pd.DataFrame(rnd, columns=['data'], index=dti)


# In[79]:


df.plot(figsize=(10, 6));
# tag: pandas_plot
# title: Pandas plot with Timestamp xticks auto-formatted


# In[80]:


pd.date_range('2016/01/01', freq='M', periods=12, tz=pytz.timezone('CET'))


# In[81]:


dti = pd.date_range('2016/01/01', freq='M', periods=12, tz='US/Eastern')
dti


# In[82]:


dti.tz_convert('GMT')
