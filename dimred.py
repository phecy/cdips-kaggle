import gensim
from gensim import corpora, models, similarities
import os
from sklearn.externals import joblib
import sys

def DimReduction(SparseMatFeatures,DR_type,NumDims):
    '''
    performs dimension reduction using gensim library
    INPUT: sparse matrix, type of reduction desired ('tfidf', 'lda','lsi'), and desired number of reduced dims 
    OUTPUT: sparse dimension reduced & normalized matrix (if 'tfidf', then not reduced but returns
            matrix where new values reflects term frequency in ad that is offset by the global frequency
            of the word in all ads)
    '''
    corpus = gensim.matutils.Sparse2Corpus(SparseMatFeatures.transpose(copy=False))
    tfidf = models.TfidfModel(corpus) 
    corpus_tfidf = tfidf[corpus]

    if (DR_type == 'tfidf'):
       scipy_csc_matrix = gensim.matutils.corpus2csc(corpus_tfidf)     
        
    if (DR_type == 'lda'):        
       lda = models.LdaModel(corpus_tfidf, num_topics=NumDims)
       corpus_lda = lsi[corpus_tfidf]  
       scipy_csc_matrix = gensim.matutils.corpus2csc(corpus_lda)          
     
    if (DR_type == 'lsi'):        
       lsi = models.LsiModel(corpus_tfidf, num_topics=NumDims)
       corpus_lsi = lsi[corpus_tfidf]
       scipy_csc_matrix = gensim.matutils.corpus2csc(corpus_lsi) 

    scipy_csc_matrix = scipy_csc_matrix.transpose(copy=True)
    return scipy_csc_matrix

def main(feature_pkl,DR_type='lsi',NumDims=100):
    # Reconstrunct features from .pkl saved by new-features.py
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
    trainFeaturesDR = DimReduction(trainFeatures,DR_type,NumDims)
    # Dump new features to same directory as source
    output_base = os.path.splitext(feature_pkl)[0]+'_'.join(DR_type,str(NumDims))+'.pkl'
    joblib.dump(trainFeaturesDR,output_base+'.pkl')

if __name__=='__main__':
    if len(sys.argv)>1:
        main(sys.argv[1:])
    else:
        print 'USAGE: python dimred.py [path-to-feature.pkl] <DR_type (tfidf,lda,lsi)> <number of dimensions/topics>'
