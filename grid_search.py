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
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(dataFolder+"\\git_folder\\new-feat-full\\"+"train_data.pkl")
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    matplotlib.rc('font', **font)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainFeatures, trainTargets, test_size=0.2, random_state=0)
    
    param_dict = {"alpha":frange(0.1,1,0.1)}
    #fit_dict = {"loss":"log","penalty":"l2","class_weight":"auto"}
    
    scores = ['precision']
    
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
    
        #clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, scoring=score)
        clf = grid_search.GridSearchCV(SGDClassifier(loss="log",penalty="l2",alpha=1e-4,class_weight="auto"),param_dict,scoring=score)
        clf.fit(X_train, y_train)
    
        print("Best parameters set found on development set:")
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() / 2, params))
        print()
    
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(metrics.classification_report(y_true, y_pred))
        print()

    logging.info("Done with grid_search")
    
                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
