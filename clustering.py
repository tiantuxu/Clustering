#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import ttest_ind_from_stats
from scipy.spatial import distance
import random
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(0)
# random.seed(0)


# In[2]:


dataset = pd.read_csv('digits-embedding.csv', header=None)
#print dataset
X = []
L = []
d = []
for i in range(10):
    current = dataset[dataset[1] == i]
    l = np.random.choice(len(current), 10)
    for j in l:
        X.append([current.iloc[j][2], current.iloc[j][3]])
        L.append(j)
        d.append([current.iloc[j][0], current.iloc[j][1], current.iloc[j][2], current.iloc[j][3]])
data = np.array(d)


# In[3]:


# def stratified_sample(arr, k):
#     a = arr[arr[:, 1].argsort()]
#     strata = np.split(a, np.where(np.diff(a[:, 1]))[0] + 1)
#     samples = [random.sample(stratum, k) for stratum in strata]
#     return np.vstack(samples)
# data = stratified_sample(np.array(dataset), 10)


# In[4]:


# X, L = data[:, 2:], data[:, 1].astype(int)
# # print X


# In[5]:


Z_single = linkage(X, 'single')
fig = plt.figure(figsize=(20, 10))
dn = dendrogram(Z_single)
plt.savefig('./dendrogram_single.png')


# In[6]:


Z_complete = linkage(X, 'complete')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z_complete)
plt.savefig('./dendrogram_complete.png')


# In[7]:


Z_average = linkage(X, 'average')
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z_average)
plt.savefig('./dendrogram_average.png')


# In[39]:


def get_stat(data, centroid, label, K):
    N = len(data)
    WC_SSD = 0
    S = []
    # d = data.iloc[:,2:4]
    data_matrix = np.array(data)
    count = 0
    for i in range(len(data)):
        # Update the WC_SSD
        WC_SSD += np.square(distance.euclidean((data[i][0], data[i][1]), centroid[label[count] - 1]))
        # Update the SC
        idx = [i for i in range(N) if label[i] == label[count]]
        # print idx
        #print idx
        SC_A = np.sum(np.sqrt(np.sum(np.square(data_matrix[idx,:] - data_matrix[count,:]), axis = 1)))
        SC_B = np.sum(np.sqrt(np.sum(np.square(data_matrix - data_matrix[count,:]), axis = 1))) - SC_A
        if len(idx) > 0:
            SC_A /= len(idx)
            SC_B /= N - len(idx)
        S.append(1.0 * (SC_B - SC_A)/max(SC_A, SC_B))
        count += 1
    
    SC = np.sum(S)/len(S)
    # print S, len(S)
    # print 'WC_SSD:', '%.2f' % WC_SSD
    # print 'SC:', '%.2f' % SC
    return WC_SSD, SC

def get_centroid(label, k):
    centroid = [[0,0] for i in range(k)]
    count = [0 for i in range(k)]
    for i in range(0, k):
        for j in range(len(label)):
            if label[j] == i+1:
                count[i] += 1
                centroid[i][0] += X[j][0]
                centroid[i][1] += X[j][1]
    # print count 
    for i in range(0, k):
        if count[i] > 0:
            centroid[i][0] /= 1.0 * count[i]
            centroid[i][1] /= 1.0 * count[i]
    # print centroid          
    return centroid
    


# In[44]:


K = [2, 4, 8, 16, 32]

WC_SSD_single = []
WC_SSD_complete = []
WC_SSD_average = []

SC_single = []
SC_complete = []
SC_average = []

for k in K:
    # print 'K:', k
    # single
    label_single = scipy.cluster.hierarchy.fcluster(Z_single, k, criterion='maxclust')
    centroid = get_centroid(label_single, k)
    WC_SSD, SC = get_stat(X, centroid, label_single, k)
    WC_SSD_single.append(WC_SSD)
    SC_single.append(SC)
    # complete
    label_complete = scipy.cluster.hierarchy.fcluster(Z_complete, k, criterion='maxclust')
    centroid = get_centroid(label_complete, k)
    WC_SSD, SC = get_stat(X, centroid, label_complete, k)
    WC_SSD_complete.append(WC_SSD)
    SC_complete.append(SC)
    # average
    label_average = scipy.cluster.hierarchy.fcluster(Z_average, k, criterion='maxclust')
    centroid = get_centroid(label_average, k)
    WC_SSD, SC = get_stat(X, centroid, label_average, k)
    WC_SSD_average.append(WC_SSD)
    SC_average.append(SC)


# In[45]:


plt.figure(figsize=(120,80))
f, axarr = plt.subplots(2, 3, figsize=(20,10))
#WC_SSD for data1
axarr[0, 0].plot(K, WC_SSD_single, marker=matplotlib.markers.CARETDOWNBASE)
axarr[0, 0].set_title('WC_SSD: single')
axarr[0, 0].set_xlabel('K')
axarr[0, 0].set_ylabel('WC-SSD')

#WC_SSD for data1
axarr[0, 1].plot(K, WC_SSD_complete, marker=matplotlib.markers.CARETDOWNBASE)
axarr[0, 1].set_title('WC_SSD: complete')
axarr[0, 1].set_xlabel('K')
axarr[0, 1].set_ylabel('WC-SSD')

#WC_SSD for data1
axarr[0, 2].plot(K, WC_SSD_complete, marker=matplotlib.markers.CARETDOWNBASE)
axarr[0, 2].set_title('WC_SSD: average')
axarr[0, 2].set_xlabel('K')
axarr[0, 2].set_ylabel('WC-SSD')

#WC_SSD for data1
axarr[1, 0].plot(K, SC_single, marker=matplotlib.markers.CARETDOWNBASE)
axarr[1, 0].set_title('SC: single')
axarr[1, 0].set_xlabel('K')
axarr[1, 0].set_ylabel('SC')

#WC_SSD for data1
axarr[1, 1].plot(K, SC_complete, marker=matplotlib.markers.CARETDOWNBASE)
axarr[1, 1].set_title('SC: complete')
axarr[1, 1].set_xlabel('K')
axarr[1, 1].set_ylabel('SC')

#WC_SSD for data1
axarr[1, 2].plot(K, SC_average, marker=matplotlib.markers.CARETDOWNBASE)
axarr[1, 2].set_title('SC: average')
axarr[1, 2].set_xlabel('K')
axarr[1, 2].set_ylabel('SC')

# Fine-tune figure; make subplots farther from each other.
f.subplots_adjust(hspace=0.2, wspace = 0.2)
plt.savefig('./3_3.png')


# In[51]:


def get_nmi(data, centroid, label, K):
    N = len(data)
    S = []
    # d = data.iloc[:,2:4]
    
    P = np.array([[0 for i in range(10)] for j in range(K)])
    H_C = [0 for i in range(K)]
    H_G = [0 for i in range(10)]
    #print P.shape
    count = 0
    for i in range(len(data)):
        # Update the information matrix
        #print index, label[index], int(row[1])
        P[label[count] - 1][int(data[i][1])] += 1
        H_C[label[count] - 1] += 1
        H_G[int(data[i][1])] += 1
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


# In[57]:


print 'single K:', 8
# single
label_single = scipy.cluster.hierarchy.fcluster(Z_single, 8, criterion='maxclust')
centroid = get_centroid(label_single, 8)
NMI_single = get_nmi(data, centroid, label_single, 8)

print 'complete K:', 8
# complete
label_complete = scipy.cluster.hierarchy.fcluster(Z_complete, 8, criterion='maxclust')
centroid = get_centroid(label_complete, 8)
NMI_single = get_nmi(data, centroid, label_complete, 8)

print 'average K:', 8
# average
label_average = scipy.cluster.hierarchy.fcluster(Z_average, 8, criterion='maxclust')
centroid = get_centroid(label_average, 8)
NMI_single = get_nmi(data, centroid, label_average, 8)

# In[ ]:




