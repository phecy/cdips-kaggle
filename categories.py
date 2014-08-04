# coding : utf-8
'''
Use pandas pivots to understand categories and json fields
'''
import ipdb
import numpy as np
import os
import pandas as pd
from scipy import sparse
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
import sys

def indicator(df,label,noncollinear=False):
    # Make csr dummy variable from categorical label
    # optionally remove last col to avoid collinearity
    dummy = pd.get_dummies(df[label])
    if noncollinear:
        dummy = dummy.drop(dummy.columns[-1])
    return sparse.csr_matrix(dummy)

def classify(dummy_train,dummy_test,feature_pkl,output_file):
    # Train classifier, iterating over subsets
    # Load Features
    print 'Loading features...'
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
    trainTargets = np.array(trainTargets)
    testItemIds = np.array(testItemIds)
    predicted_ids = []
    predicted_scores = []
    # SGD Logistic Regression per sample 
    clf = SGDClassifier(alpha=3.16227766017e-08, class_weight='auto', epsilon=0.1,
          eta0=0.0, fit_intercept=True, l1_ratio=0.15,
          learning_rate='optimal', loss='log', n_iter=5, n_jobs=1,
          penalty='elasticnet', power_t=0.5, random_state=None, shuffle=False,
          verbose=0, warm_start=False)
    for col in range(np.shape(dummy_train)[1]):
        # Get nonzero dummy indices as array
        idx_train = dummy_train[:,col].astype('bool').T.toarray()[0]
        print 'Training subset {} of {}...'.format(col,np.shape(dummy_train)[1])
        sub_train = normalize(trainFeatures.tocsr()[idx_train,:])
        clf.fit(sub_train,trainTargets[idx_train])
       # Use probabilities instead of binary class prediction in order to generate a ranking    
        idx_test = dummy_test[:,col].astype('bool').T.toarray()[0]
        sub_test = normalize(testFeatures.tocsr()[idx_test,:])
        predicted_scores += clf.predict_proba(trainTargets[idx_test]).tolist()
        predicted_ids += testItemIds[idx_test].tolist()
    
    with open(os.path.splitext(feature_pkl)[0]+'_'+output_file,'w') as out_fid:
        out_fid.write("id\n")
        for pred_score, item_id in sorted(zip(predicted_scores, predicted_ids), reverse = True):
           # only writes item_id per output spec, but may want to look at predicted_scores
            out_fid.write("%d\n" % (item_id))

def main(train_file='avito_train.tsv',test_file='avito_test.tsv',feature_pkl='test'):
   print 'Loading categories data frames...'
   df_train = pd.read_csv(train_file, sep='\t', usecols=np.array([1,2]))
   df_test = pd.read_csv(test_file, sep='\t', usecols=np.array([1,2]))
   # category
   print '_SPLIT CATEGORY_'
   # classify(indicator(df_train,'category'),indicator(df_test,'category'),feature_pkl,'category_split.csv')
   # subcategory
   print '_SPLIT SUBCATEGORY_'
   classify(indicator(df_train,'subcategory'),indicator(df_test,'subcategory'),feature_pkl,'subcategory_split.csv')
   # One classifier, but with subcategory as feature
   # print '_SUBCATEGORY AS FEATURE_'
   # indicator(df_train,'subcategory')
   # classify(np.matrix([0]),data_folder,'subcategory_feature.csv',noncollinear=True)

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
