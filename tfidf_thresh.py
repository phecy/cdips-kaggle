# coding : utf-8
'''
Threshold the neerray(tfidf_sum.tolist())gram features run on tfidf score
(eliminate unimportant features that make even sparse matrix unwieldy)
'''
from dimred import DimReduction
from extend_features import write_featureIndex
import ipdb
import matplotlib.pyplot as plt
from new_features import NEW_FEATURE_LIST
import numpy as np
import operator
import os
from scipy import sparse
from sklearn.externals import joblib
import sys

def write_hist(tfidf_sum,fname):
    plt.hist(tf_idfsum,bins=np.sqrt(len(tf_idfsum)))
    plt.savefig(fname)

def elim_exp_zeros(ngram_train):
    #Need to set explicit zeros to epsilon to avoid explicit 0 in tfidf
    #without chaging column names via prune()
    ngram_coo = ngram_train.tocoo()
    non0 = ngram_coo.data>0
    return sparse.coo_matrix((ngram_coo.data[non0],(ngram_coo.row[non0],ngram_coo.col[non0])), shape=ngram_coo.shape).tocsc()

def thresh_elim_cols(ngram_train,featureIndex,threshold=0):
    keep_idx = np.array(ngram_train.sum(axis=0).tolist()[0]) > threshold
    features,indices = zip(*sorted(featureIndex.iteritems(), key=operator.itemgetter(1)))
    nzIndex = dict((k,i) for k,i in enumerate(np.array(features)[keep_idx]))
    return ngram_train[:,keep_idx], nzIndex
   
def main(threshold,feature_pkl='Jul27-15h27m/train_data.pkl'):
    threshold = float(threshold)
    print 'Loading features pickle...'
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
    # Exclude non-ngram features
    ngram_train = trainFeatures[:,:-len(NEW_FEATURE_LIST)]
    #--
    ipdb.set_trace()
    #--
    # Eliminate explicit zeros and uniformly zero columns
    ngram_train = elim_exp_zeros(ngram_train)
    ngram_train, nzIndex = thresh_elim_cols(ngram_train,featureIndex,0)
    # Calculate TF-IDF
    tfidf = DimReduction(ngram_train,'tfidf')
    tfidf_sum = np.array(tfidf.sum(axis=0).tolist()[0])
    write_hist(tfidf_sum,'train_tfidf_hist.png')
    # threshold the columns on TF-IDF sums
    keep_idx, reducedIndex = thresh_elim_cols(tfidf_sum,nzIndex,threshold)
    # Stack the reduced features to the non-gram columns
    trainReduced = sparse.hstack(trainFeatures[:,keep_idx],trainFeatures[:,len(NEW_FEATURE_LIST):])
    testReduced = sparse.hstack(testFeatures[:,keep_idx],testFeatures[:,len(NEW_FEATURE_LIST):])
    # Add non-ngram feature labels
    end = len(reducedIndex)
    for label,i in enumerate(NEW_FEATURE_LIST):
        reducedIndex[label]=end+i
    # Write new output pkl
    out_pkl = os.path.splitext(feature_pkl)+'_tfidf-thresh.pkl'
    joblib.dump(out_pkl,(reducedIndex, trainReduced, trainTargets, trainItemIds, testReduced, testItemIds))
    print 'Writing feature names...'
    write_featureIndex(reducedIndex,os.path.splitext(feature_pkl)+'_featlist_thresh.tsv')

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
