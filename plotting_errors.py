# coding: utf-8
import csv
import re
import nltk.corpus
from collections import defaultdict
import scipy.sparse as sp
from scipy import interp
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk import SnowballStemmer
import random as rnd 
import logging
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import cross_validation
from sklearn import grid_search
import sklearn
import pdb
import datetime
import time
import pandas as pd
import sys
import matplotlib
import matplotlib.pyplot as plt

# assume data file resides in script directory
dataFolder = "C:\\Users\Cory\\Documents\\DataScienceWorkshop\\avito_kaggle\\"

#Return the model estimator function
def getmodelFunction():
    # Stochastic Gradient Descent training used (online learning)
    # loss (cost) = log ~ Logistic Regression
    # L2 norm used for cost, alpha defines learning rate
    # SGD Logistic Regression per sample 
    clf = SGDClassifier(loss="log",penalty="l2",alpha=1e-4,class_weight="auto")
    return clf

def frange(x, y, jump):
    range_list = []
    while x < y:
        range_list.append(x)
        x += jump
    return range_list
    
def main(run_name=time.strftime("%d_%H%M"), train_file="avito_train.tsv", test_file="avito_test.tsv"):
    """ Generates features and fits classifier. 
    Input command line argument is optional run name, defaults to date/time.
    """
    #Load in the .pkl data needed for fitting/cross-validation
    train_errors,test_errors = joblib.load(dataFolder+"git_folder\\new-feat-full\\errors_results.pkl")
    
    train_errors[:] = [1-x for x in train_errors]
    test_errors[:] = [1-x for x in test_errors]
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    ax.plot(frange(0.005*500, 0.2*500, 0.005*500),train_errors,color='r',label="Training Error")
    ax.plot(frange(0.005*500, 0.2*500, 0.005*500),test_errors,color='k',label="Test Error")
    ax.set_ylim([0.0,0.05])
    ax.set_xlim([0.0,100])
    plt.xlabel('Percentage of Modeling Sample')
    plt.ylabel('Error')
    legend = ax.legend(loc='upper right', shadow=True)
    
                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
