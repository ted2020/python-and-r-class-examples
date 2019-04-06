
# coding: utf-8

# In[1]:


from selenium import webdriver
import time
import numpy as np
import os
from bs4 import BeautifulSoup
import urllib.request


# # Question 1 ......................

# In[2]:


cwd=os.getcwd()
cwd


# In[3]:


root_directory = 'C:\\Users\\tuncay\\OneDrive\\PhD-CGU\\328-coding\\homeworks\\'
os.chdir(root_directory)
browser="chrome"


# In[4]:


def start_driver_windows(root_directory,chrome):
    if chrome:
        browser = webdriver.Chrome(root_directory + '/chromedriver')
    else:
        browser = webdriver.Firefox(executable_path=root_directory + '/geckodriver')
    return browser


# In[5]:


# > a) Use the function above to launch a selenium browser
k=start_driver_windows(root_directory,browser)
k


# In[6]:


# > b) Use python to programatically navigate the browser to https://www.wiley.com/en-us/subjects
k.get('https://www.wiley.com/en-us/subjects')


# In[7]:


time.sleep(np.random.uniform(1,0,1))
# > c) Once on the page, use Python to click on the link for 'Business & Management'
BM = k.find_elements_by_link_text('Business & Management') 
BM[0].click() 


# In[8]:


time.sleep(np.random.uniform(1,0,1))
# > d) On the 'Business & Management' page click on the 'VIEW ALL' link on the right hand side of the screen
BM2 = k.find_elements_by_link_text('VIEW ALL')
BM2[0].click() 


# In[9]:


time.sleep(np.random.uniform(1,0,1))
# > f) Use the dropdown menu next to 'Sort by' to select the option 'Best Seller'
BM3 = k.find_element_by_id('sortOptions-button')
BM4=BM3.send_keys("Best Seller")
BM4


# In[10]:


time.sleep(np.random.uniform(1,0,1))
# > e) Extract the HTML from the webpage of the subject you clicked on and store it in a variable called html_data

#url='https://www.wiley.com/en-us/products/Business-%26-Management-BA00?pq=%7CbestSeller%7CbestSeller%3Atrue'
#k.get("https://www.wiley.com/en-us/products/Business-%26-Management-BA00?pq=%7CbestSeller%7CbestSeller%3Atrue")

with open("html_data", "w") as p:
    p.write(k.page_source)


# In[11]:


#Once you finish the problems close the selenium browser:
k.close()


# # Question 2 .........................

# In[12]:


# a) Create a BeautifulSoup object using the HTML data you extracted in part d of Problem 1. 
a = open("html_data").read()
soup = BeautifulSoup(a)
len(soup)


# In[13]:


# b) Find all of the of the div tags of class product-content in the BeautifulSoup object

soup2=soup.find_all('div', class_='product-content')
len(soup2)

# there are 10 books in the page


# In[14]:


# c) Extract all of the text of the div tags of class product-content 
#from part b and store the text in a list called product_content
for product_content in soup2:
    print (product_content.text.replace("\n",""))

