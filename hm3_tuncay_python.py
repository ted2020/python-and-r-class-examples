
# coding: utf-8

# In[1]:


# income statement

import lxml
from lxml import html
import requests
import numpy as np
import pandas as pd

symbol = 'CMG'
#symbol=input("enter symbol, not case sensitive ==")

url = 'https://finance.yahoo.com/quote/' + symbol + '/financials?p=' + symbol

page = requests.get(url)
tree = html.fromstring(page.content)
table = tree.xpath('//table')
assert len(table) == 1

df = pd.read_html(lxml.etree.tostring(table[0], method='html'))[0]
df


# In[2]:


# balance sheet

import lxml
from lxml import html
import requests
import numpy as np
import pandas as pd

symbol = 'CMG'
#symbol=input("enter symbol, not case sensitive ==")

url = 'https://finance.yahoo.com/quote/' + symbol + '/balance-sheet?p=' + symbol

page = requests.get(url)
tree = html.fromstring(page.content)
table = tree.xpath('//table')
assert len(table) == 1

df = pd.read_html(lxml.etree.tostring(table[0], method='html'))[0]
df


# In[3]:


# cash flow

import lxml
from lxml import html
import requests
import numpy as np
import pandas as pd

symbol = 'CMG'
#symbol=input("enter symbol, not case sensitive ==")

url = 'https://finance.yahoo.com/quote/' + symbol + '/cash-flow?p=' + symbol

page = requests.get(url)
tree = html.fromstring(page.content)
table = tree.xpath('//table')
assert len(table) == 1

df = pd.read_html(lxml.etree.tostring(table[0], method='html'))[0]
df


# In[4]:


import pandas as pd
import datetime
from pandas_datareader import data as web
import matplotlib.pyplot as plt

start=datetime.datetime(2010,1,1)
#end=datetime.datetime(2019,1,1)

symbol="CMG"
#symbol=input("enter symbol, not case sensitive ==")

df=web.DataReader(symbol, "yahoo", start)
#df=web.DataReader(symbol, "yahoo", start, end)
df.tail()


# In[5]:


df['Adj Close'].plot()
plt.show()

