# coding: utf-8
"""
Modifying benchmark to:
    use Russian stemmer on description text (unused by default before)
    add feature: boolean mixedLang for correctWord() eng-rus translation
    add features: has_?, has_! (punctuation guidelines same for Russian)
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
import pandas as pd
import sys
import os

# assume data file resides in script directory
dataFolder = "C:\\Users\Cory\\Documents\\DataScienceWorkshop\\avito_kaggle\\"
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
# Original code used count threshold of 3
COUNT_MIN = 1
NEW_FEATURE_LIST = ["has_?","has_!","has_phone","has_url","has_email","has_mixed_lang","is_free"]
# From <http://www.russianlessons.net/lessons/lesson1_alphabet.php>. 33 characters, each 2 bytes
RUSSIAN_LETTERS = ur"АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
RUSSIAN_LOWER = RUSSIAN_LETTERS[1::2]
processedCnt = 0
logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)
        
def correctWord (w):
    """ Corrects word by replacing characters with written similarly depending on which language the word. 
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""
# Use a majority rule to decide if a word should be translated to all-Russian or all-English
    if len(re.findall(ur"["+RUSSIAN_LOWER+ur"]",w))>len(re.findall(ur"[a-z]",w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)                
    
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
        #words = [correctWord(w) if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(correctWord(w)) for w in cleanText.split() if len(w)>1 and w not in stopwords]
        words = [correctWord(w) if not stemmRequired else stemmer.stem(correctWord(w)) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    else:
        # Always follows else clause and breaks if only first clause run
        #words = [w if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in stopwords]
        words = [w if not stemmRequired else stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    return words

def processData(fileName, featureIndexes={}, itemsLimit=None):
    """ Processing data. """
    processMessage = ("Generate features for " if featureIndexes else "Generate features dict from ")+os.path.basename(fileName)
    logging.info(processMessage+"...")
    # This dict constructor says that when a key does not exist, add it to the dict with value 0
    wordCounts = defaultdict(lambda: 0)
    global processedCnt
    targets = []
    item_ids = []
    # Rows are examples
    row = []
    # Cols are features (featureIndexes translates words to col numbers)
    col = []
    cur_row = 0
    reader = pd.read_csv(fileName, sep='\t', header=0, chunksize=50000,encoding='utf-8')    
    for chunk in reader:
        chunk_reader = chunk.iterrows()
        for x in range(chunk.itemid.count()):
            processedCnt += 1
            item = next(chunk_reader)[1]
            if pd.isnull(item["description"]):
                item["description"] = unicode("")
            if pd.isnull(item["title"]):
                item["title"] = unicode("")
            # First call: accumulate wordCounts. Next calls: iteratively create sparse row indices
            # Defaults are no stemming and no correction
            has_mixed_lang = False
            #for word in getWords(item["title"]+" "+item["description"], stemmRequired = False, correctWordRequired = False):
            for word in getWords(item["title"]+" "+item["description"], stemmRequired = True, correctWordRequired = False):
                if not featureIndexes:
                    wordCounts[word] += 1
                else:
                    if word in featureIndexes:
                        col.append(featureIndexes[word])
                        row.append(cur_row)
                # Check for mixed Russian/English encoded words
                if not has_mixed_lang:
                    if len(re.findall(ur"["+RUSSIAN_LOWER+ur"]",word)) and len(re.findall(ur"[a-z]",word)):
                        has_mixed_lang = True
    
            # Add new feature counting / analysis with these blocks:
            if featureIndexes:
                text = item["title"]+" "+item["description"]
                if text.count("?")>0:
                      col.append(featureIndexes["has_?"])
                      row.append(cur_row)
                if text.count("!")>0:
                      col.append(featureIndexes["has_!"])
                      row.append(cur_row)
                if int(item["phones_cnt"])>0:
                      col.append(featureIndexes["has_phone"])
                      row.append(cur_row)
                if int(item["urls_cnt"])>0:
                      col.append(featureIndexes["has_url"])
                      row.append(cur_row)
                if int(item["emails_cnt"])>0:
                      col.append(featureIndexes["has_email"])
                      row.append(cur_row)
                if has_mixed_lang:
                      col.append(featureIndexes["has_mixed_lang"])
                      row.append(cur_row)
                if int(item["price"])==0:
                      col.append(featureIndexes["is_free"])
                      row.append(cur_row)
    
           # Create target (if valid) and item_id lists
            if featureIndexes:
                cur_row += 1
                if("is_blocked" in item.keys()):
                    targets.append(int(item["is_blocked"]))
                item_ids.append(int(item["itemid"]))
                        
            if processedCnt%10000 == 0:                 
                logging.debug(processMessage+": "+str(processedCnt)+" items done")
                
    if not featureIndexes:
        index = 0
        for word, count in wordCounts.iteritems():
            if count>=COUNT_MIN:
                featureIndexes[word]=index
                index += 1
        # Adding new feature indices beyond words
        for newFeature in NEW_FEATURE_LIST:
            featureIndexes[newFeature]=index
            index += 1
        return featureIndexes
    else:
        # Create spare row matrix of features -- originally 0/1 not counts
        features = sp.csr_matrix((np.ones(len(row)),(row,col)), shape=(cur_row, len(featureIndexes)), dtype=np.float64)
        if targets:
            return features, targets, item_ids
        else:
            return features, item_ids

def main(run_name=time.strftime("%d_%H%M"), train_file="avito_train.tsv", test_file="avito_test.tsv"):
    """ Generates features and fits classifier. 
    Input command line argument is optional run name, defaults to date/time.
    """
   ## This block is used to dump the feature pickle, called only once on a given train/test set. 
   ## joblib replaces standard pickle load to work well with large data objects
   ####
   # featureIndexes are words/numbers in description/title linked to sequential numerical indices
   # Note: Sampling 100 rows takes _much_ longer than using a 100-row input file
    featureIndexes = processData(dataFolder+train_file)
    # Targets refers to ads with is_blocked
   # trainFeatures is sparse matrix of [m-words x n-examples], Targets is [nx1] binary, ItemIds are ad index (for submission)
   # only ~7.6 new words (not stems) per ad. Matrix is 96.4% zeros.
    trainFeatures,trainTargets,trainItemIds = processData(dataFolder+train_file, featureIndexes)
   # Recall, we are predicting testTargets
    testFeatures,testItemIds = processData(dataFolder+test_file, featureIndexes)
    joblib.dump((trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), dataFolder+"train_data.pkl")
   ####
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(dataFolder+"train_data.pkl")
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
    output_file = "output-item-ranking.csv"
    logging.info("Writing submission to %s" % output_file)
    f = open(os.path.join(dataFolder,run_name+output_file), "w")
    f.write("id\n")
    
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
       # only writes item_id per output spec, but may want to look at predicted_scores
        f.write("%d\n" % (item_id))
    f.close()
    logging.info("Done.")
                               
if __name__=="__main__":            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+"time H:M:S = "+str(datetime.timedelta(seconds=tend-tstart))
