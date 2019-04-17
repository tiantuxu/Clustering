#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.stats import ttest_ind_from_stats
from scipy.spatial import distance
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(0)


# In[2]:


K = [2, 4, 8, 16, 32]
data1 = pd.read_csv('digits-embedding.csv', header=None)


# In[3]:


d2 = [2, 4, 6, 7]
d3 = [6, 7]
data2 = data1[data1[1].isin(d2)]
data3= data1[data1[1].isin(d3)]


# In[4]:


def kmeans(data, K):
    N = len(data)
    label = [0 for i in range(len(data))]
    count = 0
    centroid_idx = np.random.randint(0, N, size=K)
    centroid = []
    for i in centroid_idx:
        centroid.append((data.iloc[i][2], data.iloc[i][3]))
    while True:
        update = 0
        x = [0.0 for i in range(K)]
        y = [0.0 for i in range(K)]
        label_count = [0 for i in range(K)]
        idx = 0
        for index, row in data.iterrows():
            # print index, row[2], row[3]
            coor = (row[2], row[3])
            candidate_label = []
            for c in centroid:
                candidate_label.append(distance.euclidean(coor, c))
            # Get the label of the current point
            l = candidate_label.index(min(candidate_label))
            x[l] += row[2]
            y[l] += row[3]
            label_count[l] += 1
            
            # Check if there's an update in each round
            if l != label[idx]:
                label[idx] = l
                update += 1
            
            idx += 1
                
        count += 1
        
        if count == 50 or update == 0:
            break
        
        # Update the centroid if they does not meet the stop criterion
        for i in range(len(centroid)):
            if label_count[i] > 0:
                centroid[i] = (1.0 * x[i]/label_count[i], 1.0 * y[i]/label_count[i])
        
        # print count, update
    
    return label, centroid


# In[5]:


def get_stat(data, centroid, label, K):
    N = len(data)
    S = []
    d = data.iloc[:,2:4]
    
    P = np.array([[0 for i in range(10)] for j in range(K)])
    H_C = [0 for i in range(K)]
    H_G = [0 for i in range(10)]
    #print P.shape
    count = 0
    for index, row in data.iterrows():
        # Update the information matrix
        #print index, label[index], int(row[1])
        P[label[count]][int(row[1])] += 1
        H_C[label[count]] += 1
        H_G[int(row[1])] += 1
        count += 1
        
    H_C_Entropy = 0
    H_G_Entropy = 0
    MI = 0
    
    # print H_C, H_G
    for i in range(K):
        if H_C[i] != 0:
            H_C_Entropy -= (1.0 * H_C[i]/N) * np.log(1.0 * H_C[i]/N)
    for i in range(10):
        if H_G[i] != 0:
            H_G_Entropy -= (1.0 * H_G[i]/N) * np.log(1.0 * H_G[i]/N)
    for i in range(K):
        for j in range(10):
            if P[i][j] != 0:
                MI += (1.0 * P[i][j]) * np.log(1.0 * N * P[i][j]/(H_C[i] * H_G[j])) / N
    NMI = MI/(H_C_Entropy + H_G_Entropy)
    
    print 'NMI:', '%.2f' % NMI


# In[6]:


print "Dataset 1: K = 8"
np.random.seed(0)
# Dataset 1
label_1, centroid_1 = kmeans(data1, 8)
get_stat(data1, centroid_1, label_1, 8)

print "Dataset 2: K = 4"
# Dataset 2
np.random.seed(0)
label_2, centroid_2 = kmeans(data2, 4)
get_stat(data2, centroid_2, label_2, 4)

print "Dataset 3: K = 2"
# Dataset 3
np.random.seed(0)
label_3, centroid_3 = kmeans(data3, 2)
get_stat(data3, centroid_3, label_3, 2)


# In[7]:


color_choices = ['red', 'orange', 'yellow', 'green', 'black', 'blue', 'purple', 'magenta', 'gray', 'cyan']
# print data1
plt.figure()
# data1
N = len(data1)
p = np.random.randint(0, N, size=1000)
for i in range(1000):
    plt.scatter(data1.loc[p[i]][2], data1.loc[p[i]][3], c = color_choices[label_1[p[i]]])
plt.title('Cluster: Dataset 1')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
plt.savefig('./2_4-1.png')
# data2
plt.figure()
N = len(data2)
p = np.random.randint(0, N, size=1000)
for i in range(1000):
    plt.scatter(data2.iloc[p[i]][2], data2.iloc[p[i]][3], c = color_choices[label_2[p[i]]])
plt.title('Cluster: Dataset 2')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
plt.savefig('./2_4-2.png')
# data2
plt.figure()
N = len(data3)
p = np.random.randint(0, N, size=1000)
for i in range(1000):
    plt.scatter(data3.iloc[p[i]][2], data3.iloc[p[i]][3], c = color_choices[label_3[p[i]]])
plt.title('Cluster: Dataset 3')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()
plt.savefig('./2_4-3.png')


# In[ ]:




