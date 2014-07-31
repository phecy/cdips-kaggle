# coding: utf-8
from scipy import interp
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import logging
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import cross_validation
import ipdb
import datetime
import time
import sys
import matplotlib
import matplotlib.pyplot as plt

##Return the model estimator function
#def getLinearModel(loss='log',penalty='l2',alpha=1e-4,class_weight='auto'):
#    alpha=float(alpha)
#    clf = SGDClassifier(loss=loss,penalty=penalty,alpha=alpha,class_weight=class_weight)
#    print clf
#    return clf
#    
#def getEnsembleModel(n_estimators=10,max_features='auto',max_depth=None,n_jobs=-1)
#    n_estimators=int(n_estimators)
#    if max_depth is not None:
#        max_depth = int(max_depth)
#    clf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,n_jobs=n_jobs)
#    print clf
#    return clf
#
#Return the predicted class of the input test features
#def model_predicted(model,fit_features,fit_targets,test_features):
#    predicted = model.fit(fit_features, fit_targets).predict(test_features)
#    return predicted

#Return the predicted probabilities of the input test features
def model_predicted_prob(model,test_features):
    #Logistic Regression and RandomForest have predict_proba methods
    if type(model) is RandomForestClassifier or model.loss is 'log':
        return model.predict_proba(test_features).T[1]
    elif type(model) is SGDClassifier and model.loss is 'hinge':
        # Note: for SVM these are not probabilities, but decision function as orthogonal distance from margin
        return model.decision_function(test_features).T[1]
    else:
        print 'Unsupported model type'
        return -1
    
def main(feature_pkl='C:\\Users\Cory\\Documents\\DataScienceWorkshop\\avito_kaggle\\new-feat-full\\train_data.pkl', model=SGDClassifier(loss='log',penalty='l2',alpha=1e-4,class_weight='auto'), KFOLD=10):
    """ K-fold cross-validation given model and training set.
    Input path to pkl, model parameters as tuple, and number of folds
    """
    # DEFAULT MODEL:
    #    Stochastic Gradient Descent (online learning)
    #    loss (cost) = log ~ Logistic Regression
    #    L2 norm used for cost, alpha ~ Regularization
    #    class_weight = auto

    # Wrapper function may pre-load these large variables and pass as tuple instead of doing this step iteratively.
    if type(feature_pkl) is tuple: 
        featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = feature_pkl
    else:
        print 'Loading .pkl data for fitting/cross-validation...'
        featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(feature_pkl)
    if type(model) is str:
        model = eval(model)
    KFOLD = int(KFOLD)
    
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    matplotlib.rc('font', **font)
    
    # convert features to CSR for row-slicing
    trainFeatures = trainFeatures.tocsr()

    #Cross validation split into 10 folds for cross-validation
    kf_total = cross_validation.KFold(len(trainItemIds),n_folds=KFOLD,shuffle=True,indices=True)
    
    #conversion of targets to numpy 
    trainTargets = np.asarray(trainTargets)
    count = 0
    total_conf=np.zeros(shape=(2,2))
    mean_tpr = 0
    mean_fpr = np.linspace(0, 1, 100)
    
    #Iterate through the folds of the dataset
    for train_indices, test_indices in kf_total:
        #Calculation of the confusion matrix values for each fold      
        model.fit(trainFeatures[train_indices], trainTargets[train_indices])
        predicted = model.predict(trainFeatures[test_indices])
        conf_arr = metrics.confusion_matrix(trainTargets[test_indices],predicted)
        norm_conf = []        
        for i in conf_arr:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)
        total_conf += norm_conf
        #Calculation of the ROC/AUC for each fold
        prob = model_predicted_prob(model,trainFeatures[test_indices])
        fpr, tpr, thresholds = metrics.roc_curve(trainTargets[test_indices],prob)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        print "Finished with fold number " + str(count+1)
        count += 1
        
    #Calculate mean values and plot the results
    mean_tpr /= KFOLD
    mean_tpr[-1] = 1.0
    total_conf /= KFOLD
    
    #Plot the confusion matrix
    labels = ['not blocked','blocked']
    fig = plt.figure(figsize=(10,8))
    plt.clf()
    ax = fig.add_subplot(111)
    cax = ax.matshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')
    plt.title('Confusion matrix')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    #Add confusion matrix values to the graph
    width = len(norm_conf)
    height = len(norm_conf[0])
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate('%.4f' % norm_conf[x][y], xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
    print "Confusion Matrix \n" + str(total_conf)
    
    #Plot the ROC
    plt.figure(figsize=(10,8))
    plt.plot(mean_fpr,mean_tpr)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    
    auc_score = metrics.auc(mean_fpr,mean_tpr)
    print "AUC score\n" + str(auc_score)
    
    logging.info("Done with cross-validation")
    return
                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        print 'USAGE: python cross_validation.py [feature.pkl] <model_params(loss,penalty,alpha,class_weight)> <KFOLD>'
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
