# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 23:04:17 2014

@author: Cory
"""

import csv
import re
import pandas as pd
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

num_ads = 3995803 #number of ads in the training dataset
filename = 'avito_train.tsv'

samp_data = pd.read_csv(filename, sep='\t', header=0)

proved_df = samp_data[pd.notnull(samp_data['is_proved'])]

proved_df.to_csv('proved_data.tsv',sep='\t')