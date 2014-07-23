# coding : utf-8
'''
Use pandas pivots to understand categories and json fields
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

def frac_blocked(df):
   #Fraction of Blocked Ads by Category
   df_piv = pd.pivot_table(df,'subcategory','category','is_blocked',aggfunc='count')
   df_piv['frac_blocked'] = df_piv.loc[:,1]/df_piv.loc[:,0]
   df_piv = df_piv.sort(columns='frac_blocked',ascending=False)
   # translate.google.com: 
   #for ind in df_piv.index: print ind
   with open('categories_english.txt','r') as fid_eng:
       df_piv['category (ENG)'] = [c.strip().lower() for c in fid_eng]
   print df_piv

def main(in_file='avito_train.tsv'):
   # Takes 1min on full training set
   # category, subcategory, attr, is_blocked
   df = pd.read_csv(in_file, sep='\t', usecols=np.array([1,2,5,8]))
   frac_blocked(df)
   # view totals in category/subcategory
   print pd.pivot_table(df,'is_blocked','subcategory','category',aggfunc='count')

if __name__='__main__':
    if len(sys.argv)>1:
        main(sys.argv[1])
    else:
        print 'Supply input data file as tsv.'
