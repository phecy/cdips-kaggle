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
    add feature: has_json
    -- store all as float instead of logical indices (including stem counts)
'''
from collections import defaultdict
import csv
import datetime
import ipdb
import json
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
data_folder = './'
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
NEW_FEATURE_LIST = ['cnt_question','cnt_exclamation','cnt_ellipsis','cnt_phone','cnt_url','cnt_email','cnt_mixed_lang','cnt_special','frac_capital','len_ad','price','has_json']
# From <http://www.russianlessons.net/lessons/lesson1_alphabet.php>. 33 characters, each 2 bytes
RUSSIAN_LETTERS = ur'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'
RUSSIAN_LOWER = RUSSIAN_LETTERS[1::2]
LOWER_CHAR = u'a-zа-я'
UPPER_CHAR = u'A-ZА-Я'
NORMAL_CHAR = LOWER_CHAR+UPPER_CHAR+u'0-9?!.,()-/\s'
PATTERN_NONLOWER = u'[^'+LOWER_CHAR+']'
PATTERN_SPECIAL = u'[^'+NORMAL_CHAR+']'

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)

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
    with open(os.path.join(data_folder, fileName)) as items_fd:
        logging.info('Reading data...')
        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        for i, item in enumerate(itemReader):
            # After .decode(utf8), ready to use as input to stemmer
            item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
            yield i, item
    
def getWords(text, stemRequired = False, correctWordRequired = False):
    '''
    Splits the text into words, discards stop words and applies stemmer. 
    Parameters
    ----------
    text : str - initial string
    stemRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required     
    '''
    # Note: this is not a generator like getItems()
    # cleanText makes text lowercase, replaces with space if not English/Russian
    cleanText = re.sub(PATTERN_NONLOWER,' ',text.lower())
    words = [w if not stemRequired else stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    return words

def processData(fileName,featureIndex={}):
    ''' Processing data. '''
    processMessage = ('Generate features for ' if featureIndex else 'Generate features dict from ')+os.path.basename(fileName)
    logging.info(processMessage+'...')
    row,col,val = [],[],[]
    targets = []
    item_ids = []
    # Rows are examples
    # Cols are features (featureIndex translates words to col numbers)
    # Vals are counts (elements in the sparse matrix)
    cur_row = 0
    ngram_count = defaultdict(lambda: 0)
    for processedCnt, item in getItems(fileName):
        # First call: accumulate ngram_count. Next calls: iteratively create sparse row indices
        # Defaults are no stemming and no correction
        cnt_mixed_lang = 0
        # Current example's word counts: previously had global (ad corpus) ngram_count
        # This dict constructor says that when a key does not exist, add it to the dict with value 0
        ngram_count_row = defaultdict(lambda: 0)
        # Add JSON text values (checks if json module can decode)
        try:
            json_vals = ' '.join(json.loads(item['attrs']).values())
            has_json = 1
        except:
            has_json = 0
        text = ' '.join([item['title'],item['description'],json_vals])
        desc_unigram = getWords(item['description'], stemRequired = True);
        # Make 2-grams (Note: does not account for punctuation, stopwords separating sequence)
        desc_bigram = map(' '.join,zip(desc_unigram[:-1],desc_unigram[1:]))
        if not featureIndex:
            for ngram in getWords(text,stemRequired=True) + desc_bigram:
                ngram_count[ngram] += 1
        else:
            # Unigrams over title+description+json
            textStems = getWords(text,stemRequired=True)
            for ngram in set(textStems):
                if ngram in featureIndex:
                    col.append(featureIndex[ngram])
                    row.append(cur_row)
                    val.append(textStems.count(ngram))
                # Check for mixed Russian/English encoded words
                if len(re.findall(ur'[a-z]',ngram)) and len(re.findall(ur'['+RUSSIAN_LOWER+ur']',ngram)): 
                    cnt_mixed_lang += 1
            # Bigrams in description only
            for ngram in set(desc_bigram):
                if ngram in featureIndex:
                    col.append(featureIndex[ngram])
                    row.append(cur_row)
                    val.append(desc_bigram.count(ngram))

        if featureIndex:
            # Add new feature counting / analysis with these blocks:
            def append_feature(label,expr):
                row.append(cur_row)
                col.append(featureIndex[label])
                val.append(expr)
            append_feature('cnt_question',text.count('?'))
            append_feature('cnt_exclamation',text.count('!'))
            append_feature('cnt_ellipsis',text.count('...'))
            append_feature('cnt_phone',int(item['phones_cnt']))
            append_feature('cnt_url',int(item['urls_cnt']))
            append_feature('cnt_email',int(item['emails_cnt']))
            append_feature('cnt_mixed_lang',cnt_mixed_lang)
            append_feature('cnt_special',len(re.findall(PATTERN_SPECIAL,text)))
            append_feature('frac_capital',frac_capital(text))
            append_feature('len_ad',len(text))
            append_feature('price',float(item['price']))
            append_feature('has_json',has_json)

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
        # Create sparse row matrix of features -- originally 0/1 not counts
        features = sp.csr_matrix((val,(row,col)), shape=(cur_row, len(featureIndex)), dtype=np.float64)
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
   # featureIndex are words/numbers in description/title linked to sequential numerical indices
   # Note: Sampling 100 rows takes _much_ longer than using a 100-row input file
    featureIndex = processData(os.path.join(data_folder,train_file))
   # Targets refers to ads with is_blocked
   # trainFeatures is sparse matrix of [m-words x n-examples], Targets is [nx1] binary, ItemIds are ad index (for submission)
   # only ~7.6 new words (not stems) per ad. Matrix is 96.4% zeros.
    trainFeatures,trainTargets,trainItemIds = processData(os.path.join(data_folder,train_file), featureIndex)
   # Recall, we are predicting testTargets
    testFeatures,testItemIds = processData(os.path.join(data_folder,test_file), featureIndex)
    if not os.path.exists(os.path.join(data_folder,run_name)):
        os.makedirs(os.path.join(data_folder,run_name))
    joblib.dump((featureIndex, trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), os.path.join(data_folder,run_name,'train_data.pkl'))
    logging.info('Feature preparation done. Output to {}/'.format(run_name))
                               
if __name__=='__main__':            
    tstart = time.time()
    if len(sys.argv)>1:
        main(*sys.argv[1:])
    else:
        main()
    tend = time.time()
    print sys.argv[0]+'time H:M:S = '+str(datetime.timedelta(seconds=tend-tstart))
