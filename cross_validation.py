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
    
#Return the predicted class of the input test features
def model_predicted(model,fit_features,fit_targets,test_features):
    predicted = model.fit(fit_features, fit_targets).predict(test_features)
    return predicted

#Return the predicted probabilities of the input test features
def model_predicted_prob(model,fit_features,fit_targets,test_features):
    predicted_prob = model.fit(fit_features, fit_targets).predict_proba(test_features).T[1]
    return predicted_prob
    
def main(run_name=time.strftime("%d_%H%M"), train_file="avito_train.tsv", test_file="avito_test.tsv"):
    """ Generates features and fits classifier. 
    Input command line argument is optional run name, defaults to date/time.
    """
    #Load in the .pkl data needed for fitting/cross-validation
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(dataFolder+"new-feat-full\\"+"train_data.pkl")
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    matplotlib.rc('font', **font)
    
    model = getmodelFunction()
    # Stochastic Gradient Descent training used (online learning)
    # loss (cost) = log ~ Logistic Regression
    # L2 norm used for cost, alpha defines learning rate
    # SGD Logistic Regression per sample 
    
    #Cross validation split into 10 folds for cross-validation
    kf_total = cross_validation.KFold(len(trainItemIds),n_folds=10,shuffle=True,indices=True)
    
    #conversion of targets to numpy 
    trainTargets_new = np.asarray(trainTargets)
    count = 0
    total_conf=np.zeros(shape=(2,2))
    mean_tpr = 0
    mean_fpr = np.linspace(0, 1, 100)
    
    #Iterate through the folds of the dataset
    for train_indices, test_indices in kf_total:
        #Calculation of the confusion matrix values for each fold      
        predicted_values = model_predicted(model,trainFeatures[train_indices], trainTargets_new[train_indices],trainFeatures[test_indices])
        conf_arr = metrics.confusion_matrix(trainTargets_new[test_indices],predicted_values)
        norm_conf = []        
        for i in conf_arr:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)
        total_conf += norm_conf
        
        #Calculation of the ROC/AUC for each fold
        predicted_scores_prob = model_predicted_prob(model,trainFeatures[train_indices], trainTargets_new[train_indices],trainFeatures[test_indices])
        fpr, tpr, thresholds = metrics.roc_curve(trainTargets_new[test_indices],predicted_scores_prob)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        print "Finished with fold number " + str(count+1)
        count += 1
        
    #Calculate mean values and plot the results
    mean_tpr /= 10
    mean_tpr[-1] = 1.0
    total_conf /= 10
    
    #Plot the confusion matrix
    labels = ['not blocked','blocked']
    fig = plt.figure(figsize=(10,8))
    plt.clf()
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')
    plt.title('Confusion matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    #Add confusion matrix values to the graph
    width = len(norm_conf)
    height = len(norm_conf[0])
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate('%.4f' % norm_conf[x][y], xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
    print "Confusion Matrix \n" + str(total_conf)
    
    #Plot the ROC
    plt.figure(figsize=(10,8))
    plt.plot(mean_fpr,mean_tpr)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    
    auc_score = metrics.auc(mean_fpr,mean_tpr)
    print "AUC score\n" + str(auc_score)
    
    logging.info("Done with cross-validation")
    
                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
