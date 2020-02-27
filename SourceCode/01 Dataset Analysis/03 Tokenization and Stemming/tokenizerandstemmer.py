# -*- coding: utf-8 -*-
"""
Group 02

AI6122 Text Data Management and Processing
DataSet Analysis Task: Tokenization and Stemming

NTLK Tokenizer and Stemmer.

@author: Ong Jia Hui
"""
import argparse
import pandas as pd
import time
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import matplotlib.pyplot as plt
import numpy as np

import filehelper as fh

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='Tokenizer and Stemmer')
parser.add_argument('--jsonfile', type=str, default='../../../DataSet/CellPhoneReview.json',
                    help='location of json file to load as data.')
parser.add_argument('--columnname', type=str, default='reviewText',
                    help='Name of column to tokenize and stem in data.')
parser.add_argument('--lazyload', type=boolean_string, default = False,
                    help='set to true to load data from file instead of generating from NLTK. Default is False.')
parser.add_argument('--tokenfile', type=str, default='data/tokens.data',
                    help='location of the token file for lazy loading. Set empty to not generate.')
parser.add_argument('--porterfile', type=str, default='data/porter.data',
                    help='location of the porter file for lazy loading. Set empty to not generate.')
parser.add_argument('--lancasterfile', type=str, default='data/lancaster.data',
                    help='location of the lancaster file for lazy loading. Set empty to not generate.')
args = parser.parse_args()

jsonfile = args.jsonfile
if (not fh.is_not_empty_file_exists(jsonfile)):
    print("JSON file cannot be found or empty.")
    raise FileNotFoundError(jsonfile)    
colname = args.columnname
if not colname:
    raise ValueError("columnname cannot be empty.")
lazyload = True # args.lazyload
tokenfile = args.tokenfile
porterfile = args.porterfile
lancasterfile = args.lancasterfile

porter = PorterStemmer()
lancaster = LancasterStemmer()
 
data = pd.read_json(jsonfile, lines = True)

reviewtexts = list(data[colname])
tokens = []
porter_stems = []
lancaster_stems = []

#summarytexts = list(data['summary'])
#for i in range(0, len(reviewtexts)):
#    if len(reviewtexts[i]) == 0 or not reviewtexts[i]:
#        print(reviewtexts[i] + ":" + summarytexts[i])
        
if (not fh.is_not_empty_file_exists(tokenfile) or lazyload == False):
    print("Generating tokens using NLTK...")
    start_time = time.time()
    tokens = [word_tokenize(review) for review in reviewtexts]
    fh.write_list_to_file(tokenfile, tokens)  
    print("Tokenization completed in {:5.2f}ms.".format((time.time() - start_time) * 1000))
else:
    tokens = fh.load_list_from_file(tokenfile)
    print("Tokens lazily loaded from {} file.".format(tokenfile))
    
if (not fh.is_not_empty_file_exists(porterfile) or lazyload == False):
    print("Stemming tokens using Porter...")
    start_time = time.time()
    porter_stems = [[porter.stem(word) for word in tokenlist] for tokenlist in tokens]
    fh.write_list_to_file(porterfile, porter_stems)  
    print("Porter Stemming completed in {:5.2f}ms.".format((time.time() - start_time) * 1000))
else:
    porter_stems = fh.load_list_from_file(porterfile)
    print("Porter stemmed tokens lazily loaded from {} file.".format(porterfile))
    
if (not fh.is_not_empty_file_exists(lancasterfile) or lazyload == False):
    print("Stemming tokens using Lancaster...")
    start_time = time.time()
    lancaster_stems = [[lancaster.stem(word) for word in tokenlist] for tokenlist in tokens]
    fh.write_list_to_file(lancasterfile, lancaster_stems)  
    print("Lancaster Stemming completed in {:5.2f}ms.".format((time.time() - start_time) * 1000))
else:
    lancaster_stems = fh.load_list_from_file(lancasterfile)
    print("Lancaster stemmed tokens lazily loaded from {} file.".format(lancasterfile))

def generate_unique_count_xy(list_of_tokenlists):
    """ """
    unique_token_len = [len(set(tokenlist)) for tokenlist in list_of_tokenlists]
    unique_token_count = {}
    
    for tokenlen in unique_token_len:
        if not tokenlen in unique_token_count:
            unique_token_count[tokenlen] = 1
        else:
            unique_token_count[tokenlen] += 1

    tuples = sorted(unique_token_count.items()) # sorted by key, return a list of tuples
    len_of_review, no_of_reviews = zip(*tuples) # unpack a list of pairs into two tuples
    
    return len_of_review, no_of_reviews
    

token_len, token_reviews = generate_unique_count_xy(tokens)
porter_len, porter_reviews = generate_unique_count_xy(porter_stems)
# lancaster_len, lancaster_reviews = generate_unique_count_xy(lancaster_stems)

plt.style.use("seaborn-colorblind")
fig, ax = plt.subplots(2,1)

ax[0].hist([len(set(tokenlist)) for tokenlist in tokens], bins=20, histtype="step", label="tokens")
ax[0].hist([len(set(tokenlist)) for tokenlist in porter_stems], bins=20, histtype="step", label="porter stems")
ax[0].hist([len(set(tokenlist)) for tokenlist in lancaster_stems], bins=20, histtype="step", label="lancaster stems")

ax[1].bar(token_len, token_reviews, alpha=0.5, color="b", label="tokens")
ax[1].plot(token_len, token_reviews, alpha=0.5, color="b")
ax[1].bar(porter_len, porter_reviews, alpha=0.5, color="g", label="porter stems")
ax[1].plot(porter_len, porter_reviews, alpha=0.5, color="g")

ax[0].set_title('Distribution of unique words over number of reviews')
ax[0].set_ylabel('Number of reviews')
ax[1].set_ylabel('Number of reviews')
ax[1].set_xlabel('Number of unique words')
ax[0].legend()
ax[1].legend()

ax[0].grid(True)
fig.savefig("unique words vs review count.png", dpi=300)
# ax.set_xticks([0, 100, 200, 300, 400, 500])
# ax.set_xticklabels(["0", "100", "200", "300", ">=500"])
#ax.xticks([0, 100, 200, 300, 400, 500])
plt.show()