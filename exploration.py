#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.stats import ttest_ind_from_stats
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(0)


# In[2]:


raw = pd.read_csv('digits-raw.csv', header=None)
embedding = pd.read_csv('digits-embedding.csv', header=None)


# In[3]:


# os.system('mkdir figs')
for i in range(10):
    current = raw[raw[1] == i]
    l = np.random.choice(len(current), 1)
    arr = current.iloc[l].drop(current.columns[[0,1]], axis=1).values
    image = np.array(arr).reshape(28, 28)
    plt.imshow(image)
    scipy.misc.imsave('digit-' + str(i) + '.png', image)


# In[4]:


N = len(embedding)
p = np.random.randint(0, N, size=1000)
x = []
y = []
color_choices = ['red', 'orange', 'yellow', 'green', 'black', 'blue', 'purple', 'magenta', 'gray', 'cyan']
for i in range(1000):
    plt.scatter(embedding.loc[p[i]][2], embedding.loc[p[i]][3], c = color_choices[int(embedding.loc[p[i]][1])])
plt.title('Cluster')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
plt.savefig('1-2.png')


# In[ ]:




