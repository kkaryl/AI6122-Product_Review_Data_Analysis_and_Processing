import numpy as np
import gzip
import json
import pandas as pd
import os

import numpy as np
a = np.asarray([1,2,3])
b = a[::-1]
b


# https://stackoverflow.com/questions/39450065/python-3-read-write-compressed-json-objects-from-to-gzip-file

## 3 types of dict header
##['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime']
##['overall', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime']
##['overall', 'vote', 'verified', 'reviewTime', 'reviewerID', 'asin', 'style', 'reviewerName', 'reviewText', 'summary', 'unixReviewTime']
def open_file(filename):
    data = []

    for lines in gzip.open(filename, 'r'):
        line = lines.decode('UTF-8')
        line_ = json.loads(line)

        # i only pick this 5 the rest just id and format
        header = ['overall', 'reviewTime', 'reviewText', 'summary', 'unixReviewTime']
        line__ = [line_.get(h) for h in header]

        data.append(line__)

    df = pd.DataFrame(data, columns=header)
    return df

df = open_file('/') #insert location

#https://stackoverflow.com/questions/54397096/how-to-do-word-count-on-pandas-dataframe
df['length_reviewText'] = df['reviewText'].str.findall(r'(\w+)').str.len()
df['length_summary'] = df['summary'].str.findall(r'(\w+)').str.len()

df.dropna(inplace=True)
df.to_csv('/content/mydrive/My Drive/MSAI/Text Data Management and Processing/irene_text_mining.csv')
df.head()

df.shape
df['length_summary'].describe()
df['length_reviewText'].describe()
df.hist(column="length_reviewText",bins=[0,10,20,30])
df.hist(column="length_summary",bins=[0,10,20,30])

# remove z if you wan to load entire dataset.
z = 5000
text_re = " ".join(df.iloc[:z,2].values.tolist()).lower()
text_sum = " ".join(df.iloc[:z,3].values.tolist()).lower()

# https://medium.com/towards-artificial-intelligence/text-mining-in-python-steps-and-examples-78b3f8fd913b

# update nltk library
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

from nltk.corpus
    import stopwords
from nltk
    import ne_chunk

a = set(stopwords.words('english'))

# importing word_tokenize from nltk
from nltk.tokenize
    import word_tokenize
# Passing the string text into word tokenize for breaking the sentences
token = word_tokenize(text_re)

# you can remove stop word and punctuation, in here i did not remove, u need to use regex to remove cuz u can see punctuation is affecting the count
removed_stop = [w for w in token if w not in a]

# finding the frequency distinct in the tokens
# Importing FreqDist library from nltk and passing token into FreqDist
from nltk.probability
    import FreqDist
fdist = FreqDist(removed_stop)

# To find the frequency of top 10 words
fdist1 = fdist.most_common(10)
print(dist1)

# Importing Porterstemmer from nltk library
# Checking for the word ‘giving’
from nltk.stem import PorterStemmer
pst = PorterStemmer()
stemmed = [pst.stem(w) for w in removed_stop]
print(stemmed)

# Importing Lemmatizer library from nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmitized = [lemmatizer.lemmatize(w) for w in removed_stop]
print(lemmitized)

# pos tag *** chg to any lemmitized or stemmed if you wan
pos_tagged = nltk.pos_tag(removed_stop)
print(pos_tagged)
