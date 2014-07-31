# coding: utf-8
'''
Threshold the ngram features run on tfidf score
(eliminate unimportant features that make even sparse matrix unwieldy)
'''
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
from tfidf_thresh import thresh_elim_cols

def write_pkl(feature_pkl,suffix,feature_tuple):
    # Write new output pkl
    out_pkl = os.path.splitext(feature_pkl)[0]+'_'+suffix+'.pkl'
    joblib.dump((reducedIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds),out_pkl)
    print 'Writing feature names...'
    write_featureIndex(reducedIndex,os.path.splitext(feature_pkl)[0]+'_'+suffix+'_featlist.tsv')

def main(feature_pkl='Jul29-14h40m/train_data.pkl',threshold=0):
    if threshold is not None:
        threshold = float(threshold)
    print 'Loading features pickle...'
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
    print 'POOL TRAIN/TEST:'
    allFeatures = sparse.vstack((trainFeatures,testFeatures),format='csc')
    print trainFeatures.shape,testFeatures.shape,allFeatures.shape

    # Get unigram keys and indices
    keys,indices = zip(*[(k,v) for k,v in featureIndex.iteritems() if k.count(' ')==0])
    uniFeatures = allFeatures[:,indices]
    uniFeatureIndex = dict((k,i) for i,k in enumerate(keys))
    uniFeatures = uniFeatures.tocsr()
    uniTrainFeatures = uniFeatures[:trainFeatures.shape[0],:].tocsc()
    uniTestFeatures = uniFeatures[trainFeatures.shape[0]:,:].tocsc()
    print uniTrainFeatures.shape,uniTestFeatures.shape,len(uniFeatureIndex)
    write_pkl(feature_pkl,'uni',(uniFeatureIndex, uniTrainFeatures, TrainTargets, trainItemIds, uniTestFeatures, testItemIds))
    
    # Remove uniformly zero columns from all Ngrams
    ngFeatures, ngFeatureIndex, tmp = thesh_elim_cols(allFeatures,featureIndex,threshold)
    print allFeatures.shape, 'After zero removal: ', ngFeatures.shape, len(ngFeatureIndex)
    ngFeatures = ngFeatures.tocsr()
    ngTrainFeatures = ngFeatures[:trainFeatures.shape[0],:].tocsc()
    ngTestFeatures = ngFeatures[trainFeatures.shape[0]:,:].tocsc()
    print ngTrainFeatures.shape,ngTestFeatures.shape,len(ngFeatureIndex)
    write_pkl(feature_pkl,'ng',(ngFeatureIndex, ngTrainFeatures, TrainTargets, trainItemIds, ngTestFeatures, testItemIds))

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        print 'USAGE: python tfidf_thresh.py [feature_pkl] <threshold>'
