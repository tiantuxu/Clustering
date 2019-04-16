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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(0)


# In[2]:


sys.argv[1] = 'digits-embedding.csv'
sys.argv[2] = 10

data = pd.read_csv(sys.argv[1], header=None)
K = int(sys.argv[2])


# In[3]:


N = len(data)
centroid_idx = np.random.randint(0, N, size=K)
# print 'N:', N


# In[4]:


# Initial the centroid
centroid = []
for i in centroid_idx:
    centroid.append((data.loc[i][2], data.loc[i][3]))


# In[5]:


# print centroid
label = [0 for i in range(len(data))]
count = 0
while True:
    update = 0
    x = [0.0 for i in range(K)]
    y = [0.0 for i in range(K)]
    label_count = [0 for i in range(K)]
    for index, row in data.iterrows():
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
        if l != label[index]:
            label[index] = l
            update += 1
            
    count += 1
    
    if count == 50 or update == 0:
        break
    
    # Update the centroid if they does not meet the stop criterion
    for i in range(len(centroid)):
        centroid[i] = (1.0 * x[i]/label_count[i], 1.0 * y[i]/label_count[i])
    
    # print count, update


# In[6]:


WC_SSD = 0
S = []
P = [[0 for i in range(K)] for j in range(10)]
H_C = [0 for i in range(K)]
H_G = [0 for i in range(10)]
d = data.iloc[:,2:4]
data_matrix = np.array(d)


# In[7]:


for index, row in data.iterrows():
    #print index
    # Update the WC_SSD
    WC_SSD += np.square(distance.euclidean((row[2], row[3]), centroid[label[index]]))
    # Update the SC
    idx = [i for i in range(N) if label[i] == label[index]]
    #print idx
    SC_A = np.sum(np.sqrt(np.sum(np.square(data_matrix[idx,:] - data_matrix[index,:]), axis = 1)))
    SC_B = np.sum(np.sqrt(np.sum(np.square(data_matrix - data_matrix[index,:]), axis = 1))) - SC_A
    SC_A /= len(idx) - 1
    SC_B /= N - len(idx)
    S.append((SC_B-SC_A)/max(SC_A, SC_B))
    
    # Update the information matrix
    P[label[index]][int(row[1])] += 1
    H_C[label[index]] += 1
    H_G[int(row[1])] += 1
SC = np.sum(S)/len(S)
print 'WC_SSD:', '%.2f' % WC_SSD
print 'SC:', '%.2f' % SC


# In[8]:


# Get the NMI
H_C_Entropy = 0
H_G_Entropy = 0
MI = 0
# print H_C, H_G
for i in range(K):
    H_C_Entropy -= (1.0 * H_C[i]/N) * np.log(1.0 * H_C[i]/N)
for i in range(10):
    H_G_Entropy -= (1.0 * H_G[i]/N) * np.log(1.0 * H_G[i]/N)
for i in range(K):
    for j in range(10):
        if P[i][j] != 0:
            MI += (1.0 * P[i][j]) * np.log(1.0 * N * P[i][j]/(H_C[i] * H_G[j])) / N
NMI = MI/(H_C_Entropy + H_G_Entropy)
print 'NMI:', '%.2f' % NMI


# In[ ]:




