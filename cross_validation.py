# coding: utf-8
"""
Modifying benchmark to:
    use Russian stemmer on description text (unused by default before)
    add feature: boolean mixedLang for correctWord() eng-rus translation
    add features: has_?, has_! (punctuation guidelines same for Russian)
    add features: has_phone, has_url, has_email (>0 on count data)
"""
import csv
import re
import nltk.corpus
from collections import defaultdict
import scipy.sparse as sp
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
import pdb
import datetime
import time
import pandas as pd
import sys
import matplotlib.pyplot as plt

# assume data file resides in script directory
dataFolder = "C:\\Users\Cory\\Documents\\DataScienceWorkshop\\avito_kaggle\\"

def main(run_name=time.strftime("%d_%H%M"), train_file="avito_train.tsv", test_file="avito_test.tsv"):
    """ Generates features and fits classifier. 
    Input command line argument is optional run name, defaults to date/time.
    """
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(dataFolder+"new-feat-full\\"+"train_data.pkl")
    
    # Stochastic Gradient Descent training used (online learning)
    # loss (cost) = log ~ Logistic Regression
    # L2 norm used for cost, alpha defines REGULARIZATION **
    predicted_scores = []
    # SGD Logistic Regression per sample 
    clf = SGDClassifier(loss="log",penalty="l2",alpha=1e-4,class_weight="auto")
    
    #Cross validation split and calculation of accuracy
    kf_total = cross_validation.KFold(len(trainItemIds),n_folds=10,shuffle=True,indices=True)
    
   # cv = cross_validation.ShuffleSplit(len(trainItemIds), n_iter=10, train_size=0.6, random_state=0)
    trainTargets_new = np.asarray(trainTargets)
    count = 0
    fig = plt.figure(figsize=(14,10))
    plt.clf()
    for train_indices, test_indices in kf_total:
        predicted_values = clf.fit(trainFeatures[train_indices], trainTargets_new[train_indices]).predict(trainFeatures[test_indices])
        conf_arr = metrics.confusion_matrix(trainTargets_new[test_indices],predicted_values)
        norm_conf = []        
        for i in conf_arr:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)
        ax = fig.add_subplot(4,3,count)
        ax.set_aspect(1)
        ax.matshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')
        print "Finished with fold number " + str(count)
        count += 1
    #test_scores = cross_validation.cross_val_score(clf, X=trainFeatures, y=trainTargets_new, cv=kf_total,n_jobs=1)    
    #print(test_scores)
    #print("Mean Accuracy across 10-fold = " + str(test_scores.mean()))
    
    logging.info("Done with cross-validation")
    
                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
