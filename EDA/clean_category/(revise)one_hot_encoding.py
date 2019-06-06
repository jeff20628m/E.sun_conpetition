#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import graphviz
from sklearn import ensemble , preprocessing , metrics , svm , tree , linear_model
from sklearn.neural_network import MLPClassifier
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import RMSprop ,Adam


# ## import train test data

# In[88]:


data = pd.read_csv('C:\\Users\\yunting\\Documents\\GitHub\\E.sun_conpetition\\data_set\\clean_data.csv')


# In[89]:


data.head()


# In[90]:


data = data.iloc[:,1:]


# ### 將columns名稱儲存

# In[91]:


columnslist = data.columns.tolist()


# ## 儲存 building_id 來做為辨識

# In[92]:


buildid = data[columnslist[0]]


# In[93]:


buildid[:5]


# ## 將預測值 存取成train_y

# In[94]:


train_y = data[columnslist[-1]]


# In[95]:


train_y[:5]


# ## 將類別資料找出  並且以one_hot_encoding來處理

# ## 類別特徵分離出來 

# In[96]:


columnslist


# In[97]:


list_dummy = ['building_material', 
              'city', 
              'building_type', 
              'building_use', 
              'parking_way', 
              'town', 
              'village']


# In[98]:


def get_dummy(dataframe):
    dummy = pd.get_dummies(dataframe)
    return dummy


# In[99]:


building_material = get_dummy(data['building_material']) 
city = get_dummy(data['city'])
building_type = get_dummy(data['building_type']) 
building_use = get_dummy(data['building_use']) 
parking_way = get_dummy(data['parking_way'])
town = get_dummy(data['town']) 
village = get_dummy(data['village'])


# In[100]:


building_material.head()


# In[101]:


list_dummy_var = [building_material,
                  city,
                  building_type,
                  building_use,
                  parking_way,
                  town,
                  village]


# In[102]:


for item in list_dummy_var:
    print(len(item.columns))


# In[103]:


building_material_list = []
city_list = []
building_type_list = []
building_use_list = []
parking_way_list = []
town_list = []
village_list = []
for i in range(len(building_material.columns)):
    building_material_list.append('material_' + str(i))
for i in range(len(city.columns)):
    city_list.append('city_' + str(i))
for i in range(len(building_type.columns)):
    building_type_list.append('type_' + str(i))
for i in range(len(building_use.columns)):
    building_use_list.append('use_' + str(i))
for i in range(len(parking_way.columns)):
    parking_way_list.append('park_' + str(i))
for i in range(len(town.columns)):
    town_list.append('town_' + str(i))
for i in range(len(village.columns)):
    village_list.append('village' + str(i))


# In[104]:


building_material.columns = building_material_list
city.columns = city_list
building_type.columns = building_type_list
building_use.columns = building_use_list
parking_way.columns = parking_way_list
town.columns = town_list
village.columns = village_list

print(building_material.columns)

