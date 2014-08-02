# coding: utf-8
"""
Evaluate best fit model to sort avito test data.
"""
import csv
import re
import numpy as np
import operator
import os
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
import sklearn
import datetime
import time
import sys

def wordle_print(coef_rank,features):
    for coef, index in coef_rank:
        print features[index].decode('utf-8')+':'+str(coef)+'\n'

def main(feature_pkl):
    print 'Loading data...'
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
    print 'Normalizing data...'
    trainFeatures = sklearn.preprocessing.normalize(trainFeatures.tocsc(), norm='l2', axis=0)
    testFeatures = sklearn.preprocessing.normalize(testFeatures.tocsc(), norm='l2', axis=0)
    #trainSplit, testSplit = splitTuple
    # Best estimator from grid search:
    clf = SGDClassifier(alpha=1.87381742286e-07,
           class_weight='auto',
           loss='hinge',
           n_iter=10,
           penalty='l2')
    clf = SGDClassifier(alpha=3.16227766017e-08, class_weight='auto', epsilon=0.1,
       eta0=0.0, fit_intercept=True, l1_ratio=0.15,
       learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
       penalty='elasticnet', power_t=0.5, random_state=None, shuffle=False,
       verbose=0, warm_start=False)

    print 'Fitting model...'
    clf.fit(trainFeatures,trainTargets)

    # Use probabilities or decision function to generate a ranking    
    predicted_scores = clf.decision_function(testFeatures)
    with open(os.path.splitext(feature_pkl)[0]+'_testRanking.csv', 'w') as f:
        f.write('id\n')
        for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
            f.write('%d\n' % (item_id))

   # Turn estimator params into word clouds
    features, indices = zip(*sorted(featureIndex.iteritems(), key=operator.itemgetter(1)))
    coef_tuple = zip(clf.coef_[0],indices)
    coef_sort = sorted(coef_tuple, reverse=True)
    print 'Top 20 for illicit:'
    wordle_print(coef_sort[:20],features)
    print 'Top 20 for licit:'
    wordle_print(coef_sort[-20:],features)

                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        print 'USAGE: python model_eval.py [feature.pkl]'
    tend = time.time()
    print sys.argv[0]+' time H:M:S = '+str(datetime.timedelta(seconds=tend-tstart))
