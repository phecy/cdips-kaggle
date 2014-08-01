# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interp
from scipy import sparse
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler    
from sklearn.utils import shuffle
from sklearn import cross_validation
import ipdb
import datetime
import time
import sys

def main(feature_pkl):

    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)

    #Set aside 20% of train for final model selection
    trainSplit, testSplit = cross_validation.ShuffleSplit(trainFeatures.shape[0],n_iter=1,test_size=0.2)

    #Input
        #frequencies
        #TFIDF
        #LSI/LDA
    #Feature scaling
        #MinMax
        trainSplit = MinMaxScaler(trainSplit)
        #unit vector
        #z-score
    #Classifier
        #Logistic Regression
        #Linear SVM
        #Random Forest
        #Naive Bayes (Multinomial)
                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        print 'USAGE: [feature_pkl]'
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
