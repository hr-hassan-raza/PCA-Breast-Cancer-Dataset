#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[2]:


#Loading Data
cancer = load_breast_cancer()


# In[3]:


cancer.keys()


# In[4]:


df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
df.head()


# In[5]:


## Creating Labels for using in Graphs
labels = []
for i in cancer.target:
    if i == 0:
        labels.append("malignant")
    else:
        labels.append("benign")


# In[6]:


scaler = StandardScaler()
scaler.fit(df)


# In[7]:


scaled_data = scaler.transform(df)


# In[8]:


pca = PCA(n_components=2)
pca.fit(scaled_data)


# In[9]:


x_pca = pca.transform(scaled_data) ##Pca transform


# In[10]:


print("Origina Data Shape: ",scaled_data.shape)


# In[11]:


print("Reduced Data Shape: ", x_pca.shape)


# In[32]:


components = pca.components_
print("PCA component shape: ", components.shape)


# In[35]:


print("PCA components: \n",components)


# In[13]:


Xax = x_pca[:, 0] ##First Component
Yax = x_pca[:, 1] ## Second Component


# In[26]:


fig = plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
sns.scatterplot(Xax,Yax, c=cancer.target, cmap=cmap, hue=labels)

plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()


# In[21]:


from mpl_toolkits.mplot3d import Axes3D
sns.set(style = "darkgrid")

x = scaled_data[:,0]
y = scaled_data[:,1]
z = scaled_data[:,2]

fig = plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=10)
ax.scatter(x,y,z, c=cancer.target, cmap=cmap)
plt.title("First 3 features of Scaled X")
plt.show()


# In[25]:


from mpl_toolkits.mplot3d import Axes3D
sns.set(style = "darkgrid")
fig = plt.figure(figsize=(10, 8))
cmap = plt.cm.get_cmap("Spectral")
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=10, azim=10)
ax.scatter(Xax,Yax, c=cancer.target, cmap=cmap)
plt.title("First two principal components after PCA transformed")
plt.show()


# In[ ]:




