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
   
def main(threshold,feature_pkl='Jul27-15h27m/train_data.pkl'):
    print 'Loading features pickle...'
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
    #------------------------
    ipdb.set_trace()
    #------------------------
    #Exclude non-ngram features
    ngram_train = trainFeatures[:,:-len(NEW_FEATURE_LIST)]
    #Need to set explicit zeros to epsilon to avoid explicit 0 in tfidf
    #without chaging column names via prune()
    ngram_coo = ngram_train.tocoo()
    non0 = ngram_coo.data>0
    ngram_train = sparse.coo_matrix((ngram_coo.data[non0],(ngram_coo.row[non0],ngram_coo.col[non0])), shape=ngram_coo.shape).tocsc()
    ngram_train = DimReduction(ngram_train,'tfidf')
    tfidf_sum = np.array(ngram_train.sum(axis=0).tolist())
    write_hist(tfidf_sum,'train_tfidf_hist.png')

    keep_idx = tfidf_sum>threshold
    keep_arr = np.nonzero(keep_idx)
    reducedIndex = {}
    index=0
    for item in featureIndex.iteritems()
        if item[1] in keep_arr:
            reducedIndex[item[0]]=index
            index+=1
    for label in NEW_FEATURE_LIST:
        reducedIndex[label]=index
            index+=1

    trainReduced = sparse.hstack(trainFeatures[:,keep_idx],trainFeatures[:,len(NEW_FEATURE_LIST):])
    testReduced = sparse.hstack(testFeatures[:,keep_idx],testFeatures[:,len(NEW_FEATURE_LIST):])

    out_pkl = os.path.splitext(feature_pkl)+'_tfidf-thresh.pkl'
    joblib.dump(out_pkl,(reducedIndex, trainReduced, trainTargets, trainItemIds, testReduced, testItemIds))
    print 'Writing feature names...'
    write_featureIndex(reducedIndex,os.path.splitext(feature_pkl)+'_featlist_thresh.tsv')

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
