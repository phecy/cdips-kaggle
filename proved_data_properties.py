# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 13:57:01 2014

@author: Cory
"""

import csv
import re
import pandas as pd
import nltk.corpus
from collections import defaultdict
import scipy as sp
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import SGDClassifier
from nltk import SnowballStemmer
import random as rnd 
import logging
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

num_ads = 3995803 #number of ads in the training dataset
filename = "C:\\Users\\Cory\\Documents\\DataScienceWorkshop\\avito_kaggle\\proved_data.tsv"

samp_data = pd.read_csv(filename, sep='\t')

print "total number of analyzed ads = " + str(len(samp_data)) + "\n"

print "total number of legitimate ads = " + str(len(samp_data[samp_data.is_proved==1])) + "\n"

print "total number of illegitimate ads = " + str(len(samp_data[samp_data.is_proved==0])) + "\n"
 

#samp_data.groupby('is_proved').mean()
num_bins = 20;
test = samp_data.groupby("is_proved")
# Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)
plt.hist
# Create the boxplot
n, bins, patches = ax.hist([samp_data.price[samp_data.is_proved==0],samp_data.price[samp_data.is_proved==1]],bins=range(0,10000,500),normed="TRUE",color=['blue','green'])
ax.legend(["Proven Legitimate","Proven Illicit"])
fig.suptitle("Price of Item",fontsize=24)
plt.show()

#This is some code to just look at the number of data points that are extreme.
#There seem to be more extreme data points in the illicit ads, but not by a great amount (900 vs. 300 points)

#price_proved = samp_data.price[(samp_data.is_proved==1)]
#price_legit = samp_data.price[(samp_data.is_proved==0)]
#print price_proved[price_proved>10000000].describe()
#print price_legit[price_legit>10000000].describe()

#Here is just a simple t.test to see if the means of the price between illicit and legit ads
#is significant. While it provides a very low p value.  I don't think there is a real difference.
#The standard deviation of the price is just too varied.  May not make a good indicator.
price_proved = samp_data.price[(samp_data.is_proved==1)]
price_legit = samp_data.price[(samp_data.is_proved==0)]
t, prob = sp.stats.ttest_ind(price_proved[price_proved<1000000],price_legit[price_legit<1000000])
print "mean price of illegitimate = " + str(np.mean(price_proved[price_proved<1000000]))
print "std price of illegitimate = " + str(np.std(price_proved[price_proved<1000000])) + "\n"

print "mean price of legit = " + str(np.mean(price_legit[price_legit<1000000]))
print "std price of legit = " + str(np.std(price_legit[price_legit<1000000])) + "\n"

print "p-value = " + str(prob)

#Here are plots of the other numeric variables provided in the training dataset.
#The only one of interest seems to be closing time.  I think that is probably the 
#best variable of the bunch.  However, we still need to check out the categories.
fig2 = plt.figure(figsize=(9, 6))
ax2 = fig2.add_subplot(111)
n, bins, patches = ax2.hist([samp_data.emails_cnt[samp_data.is_proved==0],samp_data.emails_cnt[samp_data.is_proved==1]],bins=[0,1,2,3],normed="TRUE",color=['blue','green'])
ax2.legend(["Proven Legitimate","Proven Illicit"])
fig2.suptitle("Number of Email Addresses in Ad",fontsize=24)
plt.show()

fig3 = plt.figure(figsize=(9, 6))
ax3 = fig3.add_subplot(111)
n, bins, patches = ax3.hist([samp_data.urls_cnt[samp_data.is_proved==0],samp_data.urls_cnt[samp_data.is_proved==1]],bins=[0,1,2,3],normed="TRUE",color=['blue','green'])
ax3.legend(["Proven Legitimate","Proven Illicit"])
fig3.suptitle("Number of URLs in Ad",fontsize=24)
plt.show()

fig4 = plt.figure(figsize=(9, 6))
ax4 = fig4.add_subplot(111)
n, bins, patches = ax4.hist([samp_data.phones_cnt[samp_data.is_proved==0],samp_data.phones_cnt[samp_data.is_proved==1]],bins=[0,1,2,3],normed="TRUE",color=['blue','green'])
ax4.legend(["Proven Legitimate","Proven Illicit"])
fig4.suptitle("Number of Phone Numbers in Ad",fontsize=24)
plt.show()

fig5 = plt.figure(figsize=(9, 6))
ax5 = fig5.add_subplot(111)
n, bins, patches = ax5.hist([samp_data.close_hours[samp_data.is_proved==0],samp_data.close_hours[samp_data.is_proved==1]],bins=range(0,20,1),normed="TRUE",color=['blue','green'])
ax5.legend(["Proven Legitimate","Proven Illicit"])
fig5.suptitle("Closing Time of Ad",fontsize=24)
plt.show()

t, prob = sp.stats.ttest_ind(samp_data.close_hours[samp_data.is_proved==0],samp_data.close_hours[samp_data.is_proved==1])
print "mean closing time of illegitimate = " + str(np.mean(samp_data.close_hours[samp_data.is_proved==0]))
print "std closing time of illegitimate = " + str(np.std(samp_data.close_hours[samp_data.is_proved==0])) + "\n"

print "mean closing time of legit = " + str(np.mean(samp_data.close_hours[samp_data.is_proved==1]))
print "std closing time of legit = " + str(np.std(samp_data.close_hours[samp_data.is_proved==1])) + "\n"

print "p-value = " + str(prob)