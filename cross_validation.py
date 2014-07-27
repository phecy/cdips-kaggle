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
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
from sklearn import cross_validation
import pdb
import datetime
import time
import pandas as pd
import sys
import os

# assume data file resides in script directory
dataFolder = "C:\\Users\Cory\\Documents\\DataScienceWorkshop\\avito_kaggle\\"
# Need to run nltk.download() to get the stopwords corpus (8KB or so).
# Stop words are filtered out prior to NLP (e.g. prepositions)
#   Note: не ~ no/not and this is coded NOT to be a stop word.
        

def main(run_name=time.strftime("%d_%H%M"), train_file="avito_train.tsv", test_file="avito_test.tsv"):
    """ Generates features and fits classifier. 
    Input command line argument is optional run name, defaults to date/time.
    """
   ## This block is used to dump the feature pickle, called only once on a given train/test set. 
   ## joblib replaces standard pickle load to work well with large data objects
   ####
   # featureIndexes are words/numbers in description/title linked to sequential numerical indices
   # Note: Sampling 100 rows takes _much_ longer than using a 100-row input file
    #featureIndexes = processData(dataFolder+train_file)
    # Targets refers to ads with is_blocked
   # trainFeatures is sparse matrix of [m-words x n-examples], Targets is [nx1] binary, ItemIds are ad index (for submission)
   # only ~7.6 new words (not stems) per ad. Matrix is 96.4% zeros.
    #trainFeatures,trainTargets,trainItemIds = processData(dataFolder+train_file, featureIndexes)
   # Recall, we are predicting testTargets
    #testFeatures,testItemIds = processData(dataFolder+test_file, featureIndexes)
    #joblib.dump((trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), dataFolder+"train_data.pkl")
   ####
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(dataFolder+"new-feat-full\\"+"train_data.pkl")
    #logging.info("Feature preparation done, fitting model...")
    
    # Stochastic Gradient Descent training used (online learning)
    # loss (cost) = log ~ Logistic Regression
    # L2 norm used for cost, alpha defines learning rate
    predicted_scores = []
    # SGD Logistic Regression per sample 
    clf = SGDClassifier(loss="log",penalty="l2",alpha=1e-4,class_weight="auto")
    
    #Cross validation split and calculation of accuracy
    cv = cross_validation.ShuffleSplit(len(trainItemIds), n_iter=10, train_size=0.6, random_state=0)
    trainTargets_new = np.asarray(trainTargets)          
    test_scores = cross_validation.cross_val_score(clf, X=trainFeatures, y=trainTargets_new, cv=cv)    
    print("Accuracy: %0.4f " % (test_scores.mean()))
    logging.info("Predicting...")
   # Use probabilities instead of binary class prediction in order to generate a ranking   
    clf.fit(trainFeatures,trainTargets)
    predicted_scores = clf.predict_proba(testFeatures).T[1]
    
    logging.info("Write results...")
    output_file = "output-item-ranking.csv"
    logging.info("Writing submission to %s" % output_file)
    f = open(os.path.join(dataFolder,run_name+output_file), "w")
    f.write("id\n")
    
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
       # only writes item_id per output spec, but may want to look at predicted_scores
        f.write("%d\n" % (item_id))
    f.close()
    logging.info("Done.")
                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
