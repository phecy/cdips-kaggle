# coding: utf-8
'''
Modifying benchmark to:
    use Russian stemmer on description text (unused by default before)
        append values from JSON text to title+description text
    add feature: cnt_mixed_lang for eng-rus mixing in same word
    add features: cnt_?, cnt_!, cnt_ellipsis (punctuation guidelines same for Russian)
    add features: cnt_phone, cnt_url, cnt_email
    add feature: frac_capital: fraction of capital letters
    add feature: cnt_special: characters not in a-zA-Z/a-я/A-Я/0-9/whitespace/./?/!
    add feature: len_ad: character count of title+description
    add feature: price (raw)
    -- store all as float instead of logical indices (including stem counts)
'''
from collections import defaultdict
import csv
import datetime
import ipdb
import logging
import nltk.corpus
from nltk import SnowballStemmer
import numpy as np
import os
import random as rnd 
import re
import scipy.sparse as sp
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
import sys
import time

# assume data file resides in script directory
dataFolder = './'
# Need to run nltk.download() to get the stopwords corpus (8KB or so).
# Stop words are filtered out prior to NLP (e.g. prepositions)
#   Note: не ~ no/not and this is coded NOT to be a stop word.
stopwords= frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words('russian') if word!='не')    
# Stemming reduces inflected words to their stems (e.g. remove -ed, -ing)
stemmer = SnowballStemmer('russian')
# Some Russian letters look to the eye like English letters, but have different utf-8 codes. See: correctWord()
engChars = [ord(char) for char in u'cCyoOBaAKpPeE']
rusChars = [ord(char) for char in u'сСуоОВаАКрРеЕ']
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))
# Original code used count threshold of 3
NEW_FEATURE_LIST = ['cnt_question','cnt_exclamation','cnt_ellipsis','cnt_phone','cnt_url','cnt_email','cnt_mixed_lang','cnt_special','frac_capital','len_ad','price']
# From <http://www.russianlessons.net/lessons/lesson1_alphabet.php>. 33 characters, each 2 bytes
RUSSIAN_LETTERS = ur'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'
RUSSIAN_LOWER = RUSSIAN_LETTERS[1::2]
LOWER_CHAR = u'a-zа-я'
UPPER_CHAR = u'A-ZА-Я'
NORMAL_CHAR = u'a-zа-я0-9?!.,()-/'
PATTERN_NONLOWER = u'[^'+LOWER_CHAR+']'
PATTERN_SPECIAL = u'[^'+NORMAL_CHAR+']'

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)) 

def frac_capital(text):
    return len(re.findall(u'['+UPPER_CHAR+']',text))/float(len(text)) 

def correctWord (w):
    ''' Corrects word by replacing characters with written similarly depending on which language the word. 
        Fraudsters use this technique to avoid detection by anti-fraud algorithms.'''
# Use a majority rule to decide if a word should be translated to all-Russian or all-English
    if len(re.findall(ur'['+RUSSIAN_LOWER+ur']',w))>len(re.findall(ur'[a-z]',w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)

def getItems(fileName):
    ''' Reads data file. '''
   # This is the generator to be used by processData 
    with open(os.path.join(dataFolder, fileName)) as items_fd:
        logging.info('Sampling done. Reading data...')
        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        for i, item in enumerate(itemReader):
            # After .decode(utf8), ready to use as input to stemmer
            item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
            yield i, item
    
def getWords(text, stemmRequired = False, correctWordRequired = False):
    '''
    Splits the text into words, discards stop words and applies stemmer. 
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required     
    '''
    # Note: this is not a generator like getItems()
    # cleanText makes text lowercase, replaces with space if not English/Russian
    cleanText = re.sub(PATTERN_NONLOWER,' ',text.lower())
    words = [w if not stemmRequired else stemmer.stem(w) for w in text.lower().split() if len(w)>1 and w not in stopwords]
    return words

def processData(fileName, featureIndex = {}):
    ''' Processing data. '''
    processMessage = ('Generate features for ' if featureIndex else 'Generate features dict from ')+os.path.basename(fileName)
    logging.info(processMessage+'...')
    targets = []
    item_ids = []
    # Rows are examples
    row = []
    # Cols are features (featureIndex translates words to col numbers)
    col = []
    # Vals are counts (elements in the sparse matrix)
    val = []
    cur_row = 0
    ngram_count = defaultdict(lambda: 0)
    for processedCnt, item in getItems(fileName, itemsLimit):
        # First call: accumulate ngram_count. Next calls: iteratively create sparse row indices
        # Defaults are no stemming and no correction
        cnt_mixed_lang = 0
        # Current example's word counts: previously had global (ad corpus) ngram_count
        # This dict constructor says that when a key does not exist, add it to the dict with value 0
        ngram_count_row = defaultdict(lambda: 0)
        # Add JSON text values
        text = item['title']+' '+item['description']
        text_unigram = getWords(text, stemmRequired = True);
        text_twogram = ' '.join(zip(text_unigram[:-1],text_unigram[1:]))
        #######################################
        ipdb.set_trace()
        #######################################
        if not featureIndex:
            for ngram in text_unigram + text_twogram:
                ngram_count[ngram] += 1
        else:
            for ngram in set(getWords(text, stemmRequired = True)):
                if ngram in featureIndex:
                    col.append(featureIndex[ngram])
                    row.append(cur_row)
                    val.append(text.count(word))
                # Check for mixed Russian/English encoded words
                if len(re.findall(ur'[a-z]',ngram)) and len(re.findall(ur'['+RUSSIAN_LOWER+ur']',ngram)): 
                    cnt_mixed_lang += 1

        if featureIndex:
            # Add new feature counting / analysis with these blocks:
            def append_feature(label,expr):
                row += cur_row
                col += featureIndex[label])
                val += expr
            append_feature('cnt_question',text.count('?'))
            append_feature('cnt_exclamation',text.count('!'))
            append_feature('cnt_ellipsis',text.count('...'))
            append_feature('cnt_phone',int(item['phones_cnt']))
            append_feature('cnt_url',int(item['urls_cnt']))
            append_feature('cnt_email',int(item['emails_cnt']))
            append_feature('cnt_mixed_lang',cnt_mixed_lang)
            append_feature('cnt_special',len(re.findall(PATTERN_SPECIAL,text)))
            append_feature('frac_captial',frac_capital(text))
            append_feature('len_ad',len(text))
            append_feature('price',float(item['price']))

           # Create target (if valid) and item_id lists
            cur_row += 1
            if 'is_blocked' in item:
                targets.append(int(item['is_blocked']))
            item_ids.append(int(item['itemid']))
                    
        if processedCnt%1000 == 0:                 
            logging.debug(processMessage+': '+str(processedCnt)+' items done')

    # First call enters here, returns just the featureIndex            
    if not featureIndex:
        index = 0
        for ngram, count in ngram_count.iteritems():
            featureIndex[ngram]=index
            index += 1
        # Adding new feature indices beyond words
        for newFeature in NEW_FEATURE_LIST:
            featureIndex[newFeature]=index
            index += 1
        return featureIndex
    else:
        # Create spare row matrix of features -- originally 0/1 not counts
        features = sp.csr_matrix(val,(row,col)), shape=(cur_row, len(featureIndex)), dtype=np.float64)
        if targets:
            return features, targets, item_ids
        else:
            return features, item_ids

def main(run_name=time.strftime('%h%d-%Hh%Mm'), train_file='avito_train.tsv', test_file='avito_test.tsv'):
    ''' Generates features and fits classifier. 
    Input command line argument is optional run name, defaults to date/time.
    '''
   ## This block is used to dump the feature pickle, called only once on a given train/test set. 
   ## joblib replaces standard pickle load to work well with large data objects
   ####
   # featureIndex are words/numbers in description/title linked to sequential numerical indices
   # Note: Sampling 100 rows takes _much_ longer than using a 100-row input file
    featureIndex = processData(os.path.join(dataFolder,train_file))
    # Targets refers to ads with is_blocked
   # trainFeatures is sparse matrix of [m-words x n-examples], Targets is [nx1] binary, ItemIds are ad index (for submission)
   # only ~7.6 new words (not stems) per ad. Matrix is 96.4% zeros.
    trainFeatures,trainTargets,trainItemIds = processData(os.path.join(dataFolder,train_file), featureIndex)
   # Recall, we are predicting testTargets
    testFeatures,testItemIds = processData(os.path.join(dataFolder,test_file), featureIndex)
    if not os.path.exists(os.path.join(dataFolder,run_name)):
        os.makedirs(os.path.join(dataFolder,run_name))
    joblib.dump((featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), os.path.join(dataFolder,run_name,'train_data.pkl'))
   ####
    featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(dataFolder,run_name,'train_data.pkl'))
    logging.info('Feature preparation done, fitting model...')
    # Stochastic Gradient Descent training used (online learning)
    # loss (cost) = log ~ Logistic Regression
    # L2 norm used for cost, alpha defines learning rate
    clf = SGDClassifier(    loss='log', 
                            penalty='l2', 
                            alpha=1e-4, 
                            class_weight='auto')
    clf.fit(trainFeatures,trainTargets)

    logging.info('Predicting...')
   # Use probabilities instead of binary class prediction in order to generate a ranking    
    predicted_scores = clf.predict_proba(testFeatures).T[1]
    
    logging.info('Write results...')
    output_file = 'output-item-ranking.csv'
    logging.info('Writing submission to %s' % output_file)
    f = open(os.path.join(dataFolder,run_name,output_file), 'w')
    f.write('id\n')
    
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
       # only writes item_id per output spec, but may want to look at predicted_scores
        f.write('%d\n' % (item_id))
    f.close()
    logging.info('Done.')
                               
if __name__=='__main__':            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+'time H:M:S = '+str(datetime.timedelta(seconds=tend-tstart))
