# coding: utf-8

#import matplotlib
#matplotlib.use('Agg')
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
import sklearn.preprocessing
from sklearn.utils import shuffle
from sklearn import cross_validation
import ipdb
import datetime
import time
import sys

def minmax(mat):
    mat = mat.tocsc()
    for i in range(mat.shape[1]):
        col = mat[:,i]
        col = (col - col.min()) / (col.max() - col.min())
        mat[:,i] = col
    return mat

def print_result(clf,X_test,y_test):
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

def main(feature_pkl):
    print 'Loading training set...'
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)

    print 'Splitting training set for model selection...'
    trainSplit, testSplit, trainSplitTargets, testSplitTargets = cross_validation.train_test_split(trainFeatures, trainTargets, test_size=0.2)

#Input
    #frequencies
    #TFIDF
    #LSI/LDA
#Feature scaling
    # MinMax -- no sparse support, very slow by iteration
    #trainSplit = minmax(trainSplit)
    #testSplit = minmax(testSplit)
    #unit vector
    print 'Normalizing features (L2 unit vectors)...'
    trainSplit = sklearn.preprocessing.normalize(trainSplit.tocsc(), norm='l2', axis=0)
    testSplit = sklearn.preprocessing.normalize(testSplit.tocsc(), norm='l2', axis=0)
    #z-score
#Classifier
    #Logistic Regression and SVM with SGD
    logParams = {'loss':['hinge','log'],
            'alpha':np.logspace(-6,3,num=10).tolist(),
            'penalty':['l1','elasticnet','l2'],
            'n_iter':np.linspace(5,100,num=10).astype('int').tolist(),
            'class_weight':['auto']}
    clf_sgd = GridSearchCV(
            estimator=SGDClassifier(),
            param_grid=logParams,
            scoring='roc_auc',
            n_jobs=4,
            verbose=10,
            cv=10)
    print clf_sgd
            
    #Linear SVM
    #Random Forest
    #rfParams = {
    #'n_estimators':np.logspace(1,3,num=10).astype('int').tolist(),
    #        'criterion':('gini','entropy'), 
    #        'max_features':('sqrt','log2'),
    #        }
    #clf_rf = GridSearchCV(
    #        estimator=RandomForestClassifier(), 
    #        param_grid=rfParams, 
    #        scoring=metrics.average_precision_score,
    #        n_jobs=-1,
    #        cv=10)
    #    #Naive Bayes (Multinomial)

    for clf in (clf_sgd,):
        clf.fit(trainSplit,np.asarray(trainSplitTargets))

    print_result(clf,testSplit,testSplitTargets) 

if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        print 'USAGE: [feature_pkl]'
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
