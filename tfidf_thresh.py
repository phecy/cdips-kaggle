# coding: utf-8
'''
Threshold the ngram features run on tfidf score
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
    plt.hist(np.log10(tfidf_sum),bins=np.sqrt(len(tfidf_sum)))
    plt.xlabel('log(TFIDF)')
    plt.ylabel('Counts')
    plt.title('Training Data Importance')
    plt.savefig(fname)

def elim_exp_zeros(ngram_train):
    #Need to set explicit zeros to epsilon to avoid explicit 0 in tfidf
    #without chaging column names via prune()
    ngram_coo = ngram_train.tocoo()
    non0 = ngram_coo.data>0
    return sparse.coo_matrix((ngram_coo.data[non0],(ngram_coo.row[non0],ngram_coo.col[non0])), shape=ngram_coo.shape).tocsc()

def thresh_elim_cols(mat,featureIndex,threshold=None):
    if threshold is None:
        return mat, featureIndex, np.zeros(mat.shape[1])==0
    else:
        keep_idx = np.array(mat.sum(axis=0).tolist()[0]) > threshold
        features,indices = zip(*sorted(featureIndex.iteritems(), key=operator.itemgetter(1)))
        nzIndex = dict((k,i) for i,k in enumerate(np.array(features)[keep_idx]))
        return mat[:,keep_idx], nzIndex, keep_idx

def calc_tfidf(feat_mat,featureIndex):
    print 'Preparing features for TF-IDF...'
    # Exclude non-ngram features
    ngram_mat = feat_mat[:,:-len(NEW_FEATURE_LIST)]
    print 'Ngrams',ngram_mat.shape
    # Eliminate explicit zeros and uniformly zero columns
    ngram_mat = elim_exp_zeros(ngram_mat)
    print 'Ngrams, no explicit zeros',ngram_mat.shape
    ngram_mat, nzIndex, tmp = thresh_elim_cols(ngram_mat,featureIndex,0)
    print 'Ngrams, no uniform zero cols',ngram_mat.shape,len(nzIndex)
    # Calculate TF-IDF
    tfidf = DimReduction(ngram_mat,'tfidf')
    print 'TFIDF',tfidf.shape
    #tfidf_sum = np.array(tfidf.sum(axis=0).tolist()[0])
    #write_hist(tfidf_sum,'train_tfidf_hist.png')
    # Stack the reduced features to the non-ngram columns
    feat_mat = sparse.hstack((tfidf, feat_mat[:,-len(NEW_FEATURE_LIST):]),format='csc')
    print 'TFIDF with new features',feat_mat.shape
    return feat_mat, nzIndex

def main(feature_pkl='Jul27-15h27m/train_data.pkl',threshold=None):
    if threshold is not None:
        threshold = float(threshold)
    print 'Loading features pickle...'
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
    #______________
    ipdb.set_trace()
    #``````````````
    print 'POOL TRAIN/TEST:'
    allFeatures = sparse.vstack((trainFeatures,testFeatures),format='csc')
    print trainFeatures.shape,testFeatures.shape,allFeatures.shape
    allFeatures, reducedIndex = calc_tfidf(allFeatures,featureIndex)
    allFeatures, reducedIndex, tmp = thresh_elim_cols(allFeatures,reducedIndex,0)
    print 'TFIDF, no uniform zero cols',allFeatures.shape,len(reducedIndex)
    allFeatures = allFeatures.tocsr()
    trainFeatures = allFeatures[:trainFeatures.shape[0],:].tocsc()
    testFeatures = allFeatures[trainFeatures.shape[0]:,:].tocsc()

    #print 'TRAIN:'
    #trainFeatures, reducedIndex = calc_tfidf(trainFeatures,featureIndex)
    #print 'TEST:'
    #testFeatures, tmp = calc_tfidf(testFeatures,featureIndex)
    # Remove uniformly zero columns after TF-IDF train, keep only those cols in test
    #trainFeatures, reducedIndex, keep_idx = thresh_elim_cols(trainFeatures,reducedIndex,0)
    #testFeatures = testFeatures[:,keep_idx]

    # Add non-ngram feature labels
    end = len(reducedIndex)
    for i,label in enumerate(NEW_FEATURE_LIST):
        reducedIndex[label]=end+int(i)
    print 'Index with new features',allFeatures.shape,len(reducedIndex)
    # Write new output pkl
    out_pkl = os.path.splitext(feature_pkl)[0]+'_tfidf.pkl'
    joblib.dump((reducedIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds),out_pkl)
    print 'Writing feature names...'
    write_featureIndex(reducedIndex,os.path.splitext(feature_pkl)+'_featlist_thresh.tsv')

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        print 'USAGE: python tfidf_thresh.py [feature_pkl] <threshold>'
