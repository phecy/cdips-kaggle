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
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load("/home/temp/cdips-kaggle/Aug01-00h27m/train_data_tfidf_xprice.pkl")
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    matplotlib.rc('font', **font)
    
    #fit_dict = {"loss":"log","penalty":"l2","class_weight":"auto"}
    
    train_errors = []
    test_errors = []
    num_points = []
    
    trainSplit, testSplit, trainSplitTargets, testSplitTargets = cross_validation.train_test_split(trainFeatures, trainTargets, test_size=0.2,random_state=0)
    trainSplit = sklearn.preprocessing.normalize(trainSplit.tocsc(), norm='l2', axis=0)
    testSplit = sklearn.preprocessing.normalize(testSplit.tocsc(), norm='l2', axis=0)
    for x in frange(0.005, 0.2, 0.005):
        trainSplitnew = trainSplit[0:trainSplit.shape[0]*x]
        testSplitnew = testSplit[0:trainSplit.shape[0]*x]
        trainTargetsnew = trainSplitTargets[0:trainSplit.shape[0]*x]
        testTargetsnew = testSplitTargets[0:trainSplit.shape[0]*x]
        print("Before fitting:")
        print("x = " + str(x))
        print()
        clf = SGDClassifier(alpha=3.16227766017e-08, class_weight='auto', epsilon=0.1,
                            eta0=0.0, fit_intercept=True, l1_ratio=0.15,
                            learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
                            penalty='elasticnet', power_t=0.5, random_state=None, shuffle=False,
                            verbose=0, warm_start=False)
        clf.fit(trainSplitnew,np.asarray(trainTargetsnew))
        train_errors.append(clf.score(trainSplitnew, trainTargetsnew))
        test_errors.append(clf.score(testSplitnew, testTargetsnew))
        num_points.append(trainSplit.shape[0]*x)
        print("Fitting done, moving to next")
        print()

    logging.info("Done with grid_search")
    joblib.dump((train_errors,test_errors,num_points),"/home/temp/cdips-kaggle/errors_results.pkl")

                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
