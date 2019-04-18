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


# In[2]:


K = [2, 4, 8, 16, 32]
data1 = pd.read_csv('digits-embedding.csv', header=None)


# In[3]:


d2 = [2, 4, 6, 7]
d3 = [6, 7]
data2 = data1[data1[1].isin(d2)]
data3= data1[data1[1].isin(d3)]


# In[4]:


def kmeans(data, K, seed):
    np.random.seed(seed)
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

    return label, centroid


# In[5]:


def get_stat(data, centroid, label, K):
    N = len(data)
    WC_SSD = 0
    S = []
    d = data.iloc[:,2:4]
    data_matrix = np.array(d)
    count = 0
    for index, row in data.iterrows():
        # Update the WC_SSD
        WC_SSD += np.square(distance.euclidean((row[2], row[3]), centroid[label[count]]))
        # Update the SC
        idx = [i for i in range(N) if label[i] == label[count]]
        #print idx
        SC_A = np.sum(np.sqrt(np.sum(np.square(data_matrix[idx,:] - data_matrix[count,:]), axis = 1)))
        SC_B = np.sum(np.sqrt(np.sum(np.square(data_matrix - data_matrix[count,:]), axis = 1))) - SC_A
        SC_A /= len(idx) - 1
        SC_B /= N - len(idx)
        S.append((SC_B-SC_A)/max(SC_A, SC_B))
        count += 1
    SC = np.sum(S)/len(S)
    # print 'WC_SSD:', '%.2f' % WC_SSD
    # print 'SC:', '%.2f' % SC
    return WC_SSD, SC


# In[6]:


# Initial the centroid
# label_1 = {}
# label_2 = {}
# label_3 = {}

# centroid_1 = {}
# centroid_2 = {}
# centroid_3 = {}

WC_SSD_1 = {}
WC_SSD_2 = {}
WC_SSD_3 = {}

SC_1 = {}
SC_2 = {}
SC_3 = {}

for k in K:
    WC_SSD_1[k] = []
    WC_SSD_2[k] = []
    WC_SSD_3[k] = []

    SC_1[k] = []
    SC_2[k] = []
    SC_3[k] = []
    for s in range(10):
        # print "K =", k
        # Dataset 1
        label_1, centroid_1 = kmeans(data1, k, s)
        WC_SSD, SC = get_stat(data1, centroid_1, label_1, k)
        WC_SSD_1[k].append(WC_SSD)
        SC_1[k].append(SC)
        # Dataset 2
        label_2, centroid_2 = kmeans(data2, k, s)
        WC_SSD, SC = get_stat(data2, centroid_2, label_2, k)
        WC_SSD_2[k].append(WC_SSD)
        SC_2[k].append(SC)
        # Dataset 3
        label_3, centroid_3 = kmeans(data3, k, s)
        WC_SSD, SC = get_stat(data3, centroid_3, label_3, k)
        WC_SSD_3[k].append(WC_SSD)
        SC_3[k].append(SC)

# print WC_SSD_1
# print SC_1
# print WC_SSD_2
# print SC_2
# print WC_SSD_3
# print SC_3


# In[35]:


plt.figure(figsize=(80,40))
linestyle = {"linestyle":"-", "linewidth":4, "markeredgewidth":5, "elinewidth":5, "capsize":5}
f, axarr = plt.subplots(2, 3, figsize=(20,10))
#WC_SSD for data1
WC_SSD_1_mean = []
WC_SSD_1_stdrr = []
for k in K:
    WC_SSD_1_mean.append(np.mean(WC_SSD_1[k], axis = 0))
    WC_SSD_1_stdrr.append(np.std(WC_SSD_1[k], axis = 0)/np.sqrt(10))
#axarr[0, 0].plot(K, WC_SSD_1_mean, marker='.',color = 'b')
axarr[0, 0].errorbar(K, WC_SSD_1_mean, WC_SSD_1_stdrr, color = 'b', **linestyle)
axarr[0, 0].set_title('WC_SSD: Dataset 1')
axarr[0, 0].set_xlabel('K')
axarr[0, 0].set_ylabel('WC-SSD')

#WC_SSD for data1
WC_SSD_2_mean = []
WC_SSD_2_stdrr = []
for k in K:
    WC_SSD_2_mean.append(np.mean(WC_SSD_2[k], axis = 0))
    WC_SSD_2_stdrr.append(np.std(WC_SSD_2[k], axis = 0)/np.sqrt(10))
#axarr[0, 1].plot(K, WC_SSD_2_mean, marker='.',color = 'b')
axarr[0, 1].errorbar(K, WC_SSD_2_mean, WC_SSD_2_stdrr, color='b', **linestyle)
axarr[0, 1].set_title('WC_SSD: Dataset 2')
axarr[0, 1].set_xlabel('K')
axarr[0, 1].set_ylabel('WC-SSD')

#WC_SSD for data1
WC_SSD_3_mean = []
WC_SSD_3_stdrr = []
for k in K:
    WC_SSD_3_mean.append(np.mean(WC_SSD_3[k], axis = 0))
    WC_SSD_3_stdrr.append(np.std(WC_SSD_3[k], axis = 0)/np.sqrt(10))
#axarr[0, 2].plot(K, WC_SSD_3_mean, marker='.',color = 'b')
axarr[0, 2].errorbar(K, WC_SSD_3_mean, WC_SSD_3_stdrr, color='b', **linestyle)
axarr[0, 2].set_title('WC_SSD: Dataset 3')
axarr[0, 2].set_xlabel('K')
axarr[0, 2].set_ylabel('WC-SSD')

#WC_SSD for data1
SC_1_mean = []
SC_1_stdrr = []
for k in K:
    SC_1_mean.append(np.mean(SC_1[k], axis = 0))
    SC_1_stdrr.append(np.std(SC_1[k], axis = 0)/np.sqrt(10))
#axarr[1, 0].plot(K, SC_1_mean, marker='.',color = 'b')
axarr[1, 0].errorbar(K, SC_1_mean, SC_1_stdrr, color='b', **linestyle)
axarr[1, 0].set_title('WC_SSD: Dataset 1')
axarr[1, 0].set_xlabel('K')
axarr[1, 0].set_ylabel('SC')

#WC_SSD for data1
SC_2_mean = []
SC_2_stdrr = []
for k in K:
    SC_2_mean.append(np.mean(SC_2[k], axis = 0))
    SC_2_stdrr.append(np.std(SC_2[k], axis = 0)/np.sqrt(10))
#axarr[1, 1].plot(K, SC_2_mean, marker='.',color = 'b')
axarr[1, 1].errorbar(K, SC_2_mean, SC_2_stdrr, color='b', **linestyle)
axarr[1, 1].set_title('WC_SSD: Dataset 2')
axarr[1, 1].set_xlabel('K')
axarr[1, 1].set_ylabel('SC')

#WC_SSD for data1
SC_3_mean = []
SC_3_stdrr = []
for k in K:
    SC_3_mean.append(np.mean(SC_3[k], axis = 0))
    SC_3_stdrr.append(np.std(SC_3[k], axis = 0)/np.sqrt(10))
#axarr[1, 2].plot(K, SC_3_mean, marker='.',color = 'b')
axarr[1, 2].errorbar(K, SC_3_mean, SC_3_stdrr, color='b', **linestyle)
axarr[1, 2].set_title('WC_SSD: Dataset 3')
axarr[1, 2].set_xlabel('K')
axarr[1, 2].set_ylabel('SC')

# Fine-tune figure; make subplots farther from each other.
f.subplots_adjust(hspace=0.2, wspace = 0.2)
plt.savefig('./2_3.png')


# In[ ]:




