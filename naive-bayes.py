# coding: utf-8
"""
Modifying benchmark to:
    use Russian stemmer on description text (unused by default before)
    use bernoulli naive bayes on word occurences (instead of logistic on counts)
    add feature: boolean mixedLang for correctWord() eng-rus translation
    add features: used_exclamation, used_question (punctuation guidelines same for Russian)
    add features: has_phone, has_url, has_email (>0 on count data)
"""
import csv
import re
import nltk.corpus
from collections import defaultdict
import scipy.sparse as sp
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from nltk import SnowballStemmer
import random as rnd 
import logging
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import pdb
import datetime
import time

# assume data file resides in script directory
dataFolder = "./"
# Need to run nltk.download() to get the stopwords corpus (8KB or so).
# Stop words are filtered out prior to NLP (e.g. prepositions)
#   Note: не ~ no/not and this is coded NOT to be a stop word.
stopwords= frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian") if word!="не")    
# Stemming reduces inflected words to their stems (e.g. remove -ed, -ing)
stemmer = SnowballStemmer('russian')
# Some Russian letters look to the eye like English letters, but have different utf-8 codes. See: correctWord()
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)
        
def correctWord (w):
    """ Corrects word by replacing characters with written similarly depending on which language the word. 
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""
# Use a majority rule to decide if a word should be translated to all-Russian or all-English
    if len(re.findall(ur"[а-я]",w))>len(re.findall(ur"[a-z]",w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)

def getItems(fileName, itemsLimit=None):
    """ Reads data file. """
   # This is the generator to be used by processData 
    with open(os.path.join(dataFolder, fileName)) as items_fd:
        logging.info("Sampling...")
        # This allows for partial sampling from the input file
        if itemsLimit:
           # Iteration is a slow way to count the number of Items
            countReader = csv.DictReader(items_fd, delimiter='\t', quotechar='"')
            numItems = 0
            for row in countReader:
                numItems += 1
            items_fd.seek(0)        
            # Setting random seed makes each run of the algorithm deterministic
            rnd.seed(0)
            sampleIndexes = set(rnd.sample(range(numItems),itemsLimit))
            
        logging.info("Sampling done. Reading data...")
        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        itemNum = 0
        for i, item in enumerate(itemReader):
            item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
            # If a limit was set, then only yield the sampleIndexes items
            if not itemsLimit or i in sampleIndexes:
                itemNum += 1
                yield itemNum, item
                
    
def getWords(text, stemmRequired = False, correctWordRequired = False):
    """ Splits the text into words, discards stop words and applies stemmer. 
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required     
    """
    # Note: this is not a generator like getItems()
    # cleanText makes text lowercase, replaces with space if not English/Russian/Numeric
    cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(correctWord(w)) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    else:
        # Always follows else clause and breaks if only first clause run
        words = [w if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    
    return words

def processData(fileName, featureIndexes={}, itemsLimit=None):
    """ Processing data. """
    processMessage = ("Generate features for " if featureIndexes else "Generate features dict from ")+os.path.basename(fileName)
    logging.info(processMessage+"...")
    # This dict constructor says that when a key does not exist, add it to the dict with value 0
    wordCounts = defaultdict(lambda: 0)
    targets = []
    item_ids = []
    row = []
    col = []
    cur_row = 0
    for processedCnt, item in getItems(fileName, itemsLimit):
        #col = []
        # Defaults are no stemming and no correction
        for word in getWords(item["title"]+" "+item["description"], stemmRequired = False, correctWordRequired = False):
            if not featureIndexes:
                wordCounts[word] += 1
            else:
                if word in featureIndexes:
                    col.append(featureIndexes[word])
                    row.append(cur_row)
        
        if featureIndexes:
            cur_row += 1
            if "is_blocked" in item:
                targets.append(int(item["is_blocked"]))
            item_ids.append(int(item["itemid"]))
                    
        if processedCnt%1000 == 0:                 
            logging.debug(processMessage+": "+str(processedCnt)+" items done")
                
    if not featureIndexes:
        index = 0
        for word, count in wordCounts.iteritems():
            if count>=3:
                featureIndexes[word]=index
                index += 1
                
        return featureIndexes
    else:
        features = sp.csr_matrix((np.ones(len(row)),(row,col)), shape=(cur_row, len(featureIndexes)), dtype=np.float64)
        if targets:
            return features, targets, item_ids
        else:
            return features, item_ids

def main():
    """ Generates features and fits classifier. """
   ## This block is used to dump the feature pickle, called only once on a given train/test set. 
   ## joblib replaces standard pickle load to work well with large data objects
   ####
   # featureIndexes are words/numbers in description/title linked to sequential numerical indices
   # Note: Sampling 100 rows takes _much_ longer than using a 100-row input file
    featureIndexes = processData(os.path.join(dataFolder,"avito_train.tsv"))
    # Targets refers to ads with is_blocked
    ###############
    #pdb.set_trace()
    ###############
   # trainFeatures is sparse matrix of [m-words x n-examples], Targets is [nx1] binary, ItemIds are ad index (for submission)
   # only ~7.6 new words (not stems) per ad. Matrix is 96.4% zeros.
    trainFeatures,trainTargets,trainItemIds = processData(os.path.join(dataFolder,"avito_train.tsv"), featureIndexes)
   # Recall, we are predicting testTargets
    testFeatures,testItemIds = processData(os.path.join(dataFolder,"avito_test.tsv"), featureIndexes)
    joblib.dump((trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), os.path.join(dataFolder,"train_data.pkl"))
   ####
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(dataFolder,"train_data.pkl"))
    logging.info("Feature preparation done, fitting model...")
    # Stochastic Gradient Descent training used (online learning)
    # loss (cost) = log ~ Logistic Regression
    # L2 norm used for cost, alpha defines learning rate
    clf = SGDClassifier(    loss="log", 
                            penalty="l2", 
                            alpha=1e-4, 
                            class_weight="auto")
    clf.fit(trainFeatures,trainTargets)

    logging.info("Predicting...")
   # Use probabilities instead of binary class prediction in order to generate a ranking    
    predicted_scores = clf.predict_proba(testFeatures).T[1]
    
    logging.info("Write results...")
    output_file = "avito_starter_solution.csv"
    logging.info("Writing submission to %s" % output_file)
    f = open(os.path.join(dataFolder,output_file), "w")
    f.write("id\n")
    
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
       # only writes item_id per output spec, but may want to look at predicted_scores
        f.write("%d\n" % (item_id))
    f.close()
    logging.info("Done.")
                               
if __name__=="__main__":            
    tstart = time.time()
    main()
    tend = time.time()
    print "benchmark_avito.py time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
