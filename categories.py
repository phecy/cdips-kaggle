# coding : utf-8
'''
Use pandas pivots to understand categories and json fields
'''
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

def classify(dummy_train,dummy_test,data_folder,output_file):
    # Train classifier, iterating over subsets
    # Load Features
    print 'Loading features...'
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(data_folder,"train_data.pkl"))
    trainTargets = np.array(trainTargets)
    testItemIds = np.array(testItemIds)
    predicted_ids = []
    predicted_scores = []
    # SGD Logistic Regression per sample 
    clf = SGDClassifier(loss="hinge",penalty="l2",alpha=1e-4,class_weight="auto")
    for col in range(np.shape(dummy_train)[1]):
        # Get nonzero dummy indices as array
        idx_train = dummy_train[:,col].astype('bool').T.toarray()[0]
        print 'Training subset {} of {}...'.format(col,np.shape(dummy_train)[1])
        clf.fit(trainFeatures[idx_train,:],trainTargets[idx_train])
       # Use probabilities instead of binary class prediction in order to generate a ranking    
        idx_test = dummy_test[:,col].astype('bool').T.toarray()[0]
        predicted_scores += clf.predict_proba(testFeatures[idx_test,:]).T[1].tolist()
        predicted_ids += testItemIds[idx_test].tolist()
    
    with open(os.path.join(data_folder,output_file),'w') as out_fid:
        out_fid.write("id\n")
        for pred_score, item_id in sorted(zip(predicted_scores, predicted_ids), reverse = True):
           # only writes item_id per output spec, but may want to look at predicted_scores
            out_fid.write("%d\n" % (item_id))

def main(train_file='avito_train.tsv',test_file='avito_test.tsv',data_folder='new-feat-full/'):
   print 'Loading categories data frames...'
   df_train = pd.read_csv(train_file, sep='\t', usecols=np.array([1,2]))
   df_test = pd.read_csv(test_file, sep='\t', usecols=np.array([1,2]))
   # category
   print '_SPLIT CATEGORY_'
   classify(indicator(df_train,'category'),indicator(df_test,'category'),data_folder,'category_split_svm.csv')
   # subcategory
   print '_SPLIT SUBCATEGORY_'
   classify(indicator(df_train,'subcategory'),indicator(df_test,'subcategory'),data_folder,'subcategory_split_svm.csv')
   # One classifier, but with subcategory as feature
   # print '_SUBCATEGORY AS FEATURE_'
   # indicator(df_train,'subcategory')
   # classify(np.matrix([0]),data_folder,'subcategory_feature.csv',noncollinear=True)

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
