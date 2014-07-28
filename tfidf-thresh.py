# coding : utf-8
'''
Add the following features to the new-feat-full/train_data.pkl set:
   
    has_dummy_price (null | zero | one)
    category, subcategory as dummy variables
        drop last for noncollinearity
    price*category/subcategory cross-features. 
    (word cross features -- use logic then map to float)
    -- feature scaling: unit vector, z-score, range scaling
        This will scale price to individual categories via cross features

'''
import ipdb
import numpy as np
import operator
import os
import pandas as pd
from scipy import sparse
from sklearn.externals import joblib
import sys

def write_featureIndex(featureIndex,out_filename):
    with open(out_filename,'w') as out_fid:
        for feature in sorted(featureIndex.iteritems(), key=operator.itemgetter(1)):
            out_fid.write(feature[0]+'\t'+str(feature[1])+'\n')

def indicator(df,label,noncollinear=False):
    # Make csr dummy variable from categorical label
    # optionally remove last col to avoid collinearity
    dummy = pd.get_dummies(df[label])
    if noncollinear:
        dummy = dummy.drop(dummy.columns[-1])
    return sparse.csr_matrix(dummy), dummy.labels

def dummy_price_cross(df,label,price):
   # Return (sparse matrix of indicator - indicator*price data, labels for this data)
   sp_dummy,dummy_label = indicator(df,label,noncollinear=True)
   new_label = []
   #Add cross labels
   for dl in dummy_label:
       new_label.append(dl+'*price')
   #elementwise multiply price with indicator
   sp_cross = sp_dummy.multiply(np.tile(price,shape(sp_dummy)[1]).T)
   return sparse.hstack(sp_dummy,sp_cross), dummy_label+new_label

def main(train_file='avito_train.tsv',test_file='avito_test.tsv',feature_pkl='Jul27-15h27m/train_data.pkl'):
   print 'Loading features pickle...'
   featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
   #------------------------
   ipdb.set_trace()
   #------------------------
   print 'Writing feature names...'
   write_featureIndex(featureIndex,os.path.splitext(feature_pkl)+'_featlist.tsv')
   print 'Loading categories data frames...'
   df_test = pd.read_csv(test_file, sep='\t', usecols=np.array([1,2]))
   for feat_mat,source_file in zip((trainFeatures,testFeatures),(train_file,test_file)):
       df = pd.read_csv(source_file, sep='\t', usecols=np.array([1,2]))
       for label in ('category','subcategory'):
           dpc_sp,dpc_label = dummy_price_cross(df, label, feat_mat[:,featureIndex['price']].toarray())
           feat_mat = sparse.hstack(feat_mat,dpc_sp)
           featureIndex += dpc_label
   out_pkl = os.path.splitext(feature_pkl)+'_xprice.pkl'
   joblib.dump(out_pkl,(featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds))

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
