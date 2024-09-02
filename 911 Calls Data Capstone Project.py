#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Capstone Project

# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# 

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("911.csv")


# In[4]:


df.info()


# In[5]:


df.head()


# ## Basic Questions

# ** What are the top 5 zipcodes for 911 calls? **

# In[6]:


df["zip"].value_counts().head()


# ** What are the top 5 townships (twp) for 911 calls? **

# In[7]:


df["twp"].value_counts().head()


# In[8]:


len(df["title"].unique())


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Using .apply() with a custom lambda expression for creating a new column called "Reason" that contains this string value.** 
# 

# In[9]:


df["reason"]= df["title"].apply(lambda title : title.split(':')[0])
df.head()


# **  The most common Reason for a 911 call based off of this new column? **

# In[10]:


df["reason"].value_counts()


# ** Now using  seaborn to create a countplot of 911 calls by Reason. **

# In[11]:


sns.countplot(data = df, x ="reason")


# ___
# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[12]:


df["timeStamp"].iloc[0]


# ** You should have seen that these timestamps are still strings. 

# In[13]:


df["timeStamp"]=pd.to_datetime(df["timeStamp"])
df.info()


# ** Creating three new column for hour, month, day of week for further insights **

# In[14]:


dmap = {1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}
df["hour"] = df["timeStamp"].apply(lambda time: time.hour)
df["month"] = df["timeStamp"].apply(lambda time: time.month)
df["day of week"] = df["timeStamp"].apply(lambda time: time.dayofweek)
df["day of week"]=df["day of week"].map(dmap)


# In[15]:


df.head()
df["month"].value_counts().sort_index()


# ** Now using seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[16]:


sns.countplot(data = df,x ="day of week",hue ="reason",palette='viridis')
plt.legend( bbox_to_anchor=(1.05, 1),loc =2,borderaxespad=0.)


# **Now doing the same for Month:**

# In[17]:


sns.countplot(data = df,x ="month",hue ="reason",palette='viridis')
plt.legend( bbox_to_anchor=(1.05, 1),loc =2,borderaxespad=0.)


# ** we can see that some data are missing in the month column, so we have to fill this null value**

# In[18]:


bymonth = df.groupby("month").count()
bymonth


# ** Now creating a simple plot off of the dataframe indicating the count of calls per month. **

# In[19]:


bymonth["lat"].plot()


# ** Now see if we  can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[20]:


sns.countplot(data = df,x ="month",palette='viridis')
plt.legend( bbox_to_anchor=(1.05, 1),loc =2,borderaxespad=0.)


# In[21]:


sns.lmplot(data = bymonth.reset_index("month"),x ="month", y = "twp")


# **Creating a new column called 'Date' that contains the date from the timeStamp column. we will need to use apply along with the .date() method. ** 

# In[22]:


df["date"] = df["timeStamp"].apply(lambda timeStamp: timeStamp.date())
df.head()


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[23]:


bydate = df.groupby("date").count()
bydate.head()
bydate["lat"].plot()
plt.tight_layout()


# ** Now recreating this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[24]:


df[df["reason"]=="Traffic"].groupby("date").count()["lat"].plot()
plt.title("traffic")

plt.tight_layout()


# In[25]:


df[df["reason"]=="Fire"].groupby("date").count()["lat"].plot()
plt.title("Fire")
plt.tight_layout()


# In[26]:


df[df["reason"]=="EMS"].groupby("date").count()["lat"].plot()
plt.title("EMS")
plt.tight_layout()


# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack]

# In[27]:


df1=df.groupby(["day of week","hour"]).count()['reason'].unstack()
df1


# In[28]:


sns.heatmap(df1,cmap="viridis")


# In[29]:


sns.clustermap(data = df1,cmap="viridis")


# ** Now repeating these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[30]:


dfmonth = df.groupby(["day of week","month"]).count()["reason"].unstack()
dfmonth


# In[31]:


sns.heatmap(data = dfmonth, cmap="viridis")


# In[32]:


sns.clustermap(data = dfmonth,cmap = "coolwarm")


# In[ ]:




