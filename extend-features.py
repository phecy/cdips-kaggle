# coding : utf-8
'''
Add the following features to the new-feat-full/train_data.pkl set:
   
    price_null (OR zero OR one)
    price (raw)
    price*category/subcategory cross-features. 
        use category/subcategory as dummy variables
            drop last for noncollinearity
    (word cross features -- use logic then map to float)
    -- feature scaling: z-score (also try range scaling)
        This will scale price to individual categories via cross features

'''
import ipdb
import numpy as np
import os
import pandas as pd
from scipy import sparse
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
import sys

def indicator(df,label,noncollinear=False):
    # Make csr dummy variable from categorical label
    # optionally remove last col to avoid collinearity
    dummy = pd.get_dummies(df[label])
    if noncollinear:
        dummy = dummy.drop(dummy.columns[-1])
    return sparse.csr_matrix(dummy)

def main(train_file='avito_train.tsv',test_file='avito_test.tsv',data_folder='new-feat-full/'):
   print 'Loading new-features pickle...'
   trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(data_folder,"train_data.pkl"))
   print 'Loading categories data frames...'
   df_train = pd.read_csv(train_file, sep='\t', usecols=np.array([1,2]))
   df_test = pd.read_csv(test_file, sep='\t', usecols=np.array([1,2]))
   # category
   indicator(df_train,'category',True)
   indicator(df_test,'category',True)
   # append labels
   # subcategory
   indicator(df_train,'subcategory',True)
   indicator(df_test,'subcategory',True)

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
