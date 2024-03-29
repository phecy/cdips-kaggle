# coding: utf-8
'''
Use BernoulliNB on saved featureset joblib pkls
'''
import scipy.sparse as sp
import numpy as np
import os
from sklearn.naive_bayes import BernoulliNB
import logging
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import pdb
import datetime
import time
import sys

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)

def main(output_file=time.strftime('%h%d-%Hh%Mm')+'.csv', in_pkl=None):
    """ Generates features and fits classifier. 
    Input command line argument is optional run name, defaults to date/time.
    """
    logging.info("Loading features...")
    if not in_pkl:
        return "input .plk required"
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(in_pkl)
    logging.info("Loaded features, fitting model...")
    # Bernoulli Naive Bayes
    clf = BernoulliNB(alpha=1.0, binarize=None, fit_prior=True)
    clf.fit(trainFeatures,trainTargets)
    logging.info("Predicting...")
    # Use probabilities instead of binary class prediction in order to generate a ranking    
    predicted_scores = clf.predict_log_proba(testFeatures).T[1]

    logging.info("Write results...")
    logging.info("Writing submission to %s" % output_file)
    f = open(output_file, "w")
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
        print "Not enough args"
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
