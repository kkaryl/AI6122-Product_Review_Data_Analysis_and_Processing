import nltk
import pandas as pd
import gzip
import os
import json
import numpy as np
from yellowbrick.text import PosTagVisualizer
nltk.download
from nltk import tokenize
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag

#Import Stemmers
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

stemmer_porter = PorterStemmer()
stemmer_lancaster = LancasterStemmer()
stemmer_snowball = SnowballStemmer('english')

#Load Reviews
def open_file(filename):
    data = []
    for line in open(filename, 'r'):
        data_ = json.loads(line)
        header = ['overall', 'reviewTime', 'reviewText', 'summary', 'unixReviewTime']
        line__ = [data_.get(h) for h in header]
        data.append(line__)
        df = pd.DataFrame(data, columns=header)
    return df

df = open_file('Cell_Phones_and_Accessories_5.json')
text_list = df['reviewText'].values.tolist()Pandas library

ttagged_list = []
ptagged_list = []
ltagged_list = []
stagged_list = []

for text in text_list:
    tokens = nltk.word_tokenize(text)
    portstemmed = [stemmer_porter.stem(token) for token in tokens]
    lancasterstemmed = [stemmer_lancaster.stem(token) for token in tokens]
    SBstemmed = [stemmer_snowball.stem(token) for token in tokens]
    ptagged = nltk.pos_tag(portstemmed)
    ltagged = nltk.pos_tag(lancasterstemmed)
    stagged = nltk.pos_tag(SBstemmed)
    ptagged_list.append(ptagged)
    ltagged_list.append(ltagged)
    stagged_list.append(stagged)

ptagged_list = np.expand_dims(np.asarray(ptagged_list),0)
ltagged_list = np.expand_dims(np.asarray(ltagged_list),0)
stagged_list = np.expand_dims(np.asarray(stagged_list),0)
print (ptagged_list.shape)
print (ltagged_list.shape)
print (stagged_list.shape)

# Create the visualizer 
vizp = PosTagVisualizer()
vizp.fit(ptagged_list)
vizp.show('/Users/irenengyusi/Desktop/Text Data Assignment/Porter.png')

vizl= PosTagVisualizer()
vizl.fit(ltagged_list)
vizl.show('/Users/irenengyusi/Desktop/Text Data Assignment/Lancaster.png')

vizs = PosTagVisualizer()
vizs.fit(stagged_list)
vizs.show('/Users/irenengyusi/Desktop/Text Data Assignment/Snowball.png')
