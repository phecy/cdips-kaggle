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

def train(dummy,data_folder,output_file):
    # Train classifier, iterating over subsets
    # Load Features
    print 'Loading features...'
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(data_folder,"train_data.pkl"))
    predicted_scores = []
    predicted_ids = []
    # SGD Logistic Regression per sample 
    clf = SGDClassifier(loss="log",penalty="l2",alpha=1e-4,class_weight="auto")
    for col in range(np.shape(dummy)[1]):
        # Get nonzero dummy indices as array
        idx = dummy[:,col].astype('bool').T.toarray()[0]
        print 'Training subset {} of {}...\n'.format(col,np.shape(dummy)[1])
        clf.fit(trainFeatures[idx,:],trainTargets[idx,:])
       # Use probabilities instead of binary class prediction in order to generate a ranking    
        predicted_scores += list(clf.predict_proba(testFeatures[idx,:]).T[1])
        predicted_ids += testItemIds[idx,:]
    
    with open(os.path.join(data_folder,output_file),'w') as out_fid:
        out_fid.write("id\n")
        for pred_score, item_id in sorted(zip(predicted_scores, predicted_ids), reverse = True):
           # only writes item_id per output spec, but may want to look at predicted_scores
            out_fid.write("%d\n" % (item_id))

def main(in_file='avito_train.tsv',data_folder='new-feat-full/'):
   print 'Loading categories data frame...'
   df = pd.read_csv(in_file, sep='\t', usecols=np.array([1,2]))
   # category
   print '_SPLIT CATEGORY_'
   train(indicator(df,'category'),data_folder,'category_split.csv')
   # subcategory
   print '_SPLIT SUBCATEGORY_'
   train(indicator(df,'subcategory'),data_folder,'subcategory_split.csv')
   # One classifier, but with subcategory as feature
   # print '_SUBCATEGORY AS FEATURE_'
   # indicator(df,'subcategory')
   # train(np.matrix([0]),data_folder,'subcategory_feature.csv',noncollinear=True)

if __name__=='__main__':
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
