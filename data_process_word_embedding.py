"""
Code to process text data, build the word embeddings, and ready it for NLP pipeline 
"""
import pandas as pd
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re

# First load the data in, and we only need the text from the files, so we load in column 1, skip the first header row and use the custom header of (text) as it's just a bit more descriptive
subreddit_0 = pd.read_csv('./data/0_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_1 = pd.read_csv('./data/1_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_2 = pd.read_csv('./data/2_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_3 = pd.read_csv('./data/3_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_4 = pd.read_csv('./data/4_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_5 = pd.read_csv('./data/5_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_6 = pd.read_csv('./data/6_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_7 = pd.read_csv('./data/7_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_8 = pd.read_csv('./data/8_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_9 = pd.read_csv('./data/9_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_10 = pd.read_csv('./data/10_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])
subreddit_11 = pd.read_csv('./data/11_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1], lineterminator='\n') # There was something weird about the characters in this file that threw an overflow error. Potentially characters of the form '\r'. So just added a line_terminator qualifier
subreddit_12 = pd.read_csv('./data/12_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])   
subreddit_13 = pd.read_csv('./data/13_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])   
subreddit_14 = pd.read_csv('./data/14_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])   
subreddit_15 = pd.read_csv('./data/15_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])   
subreddit_16 = pd.read_csv('./data/16_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])   
subreddit_17 = pd.read_csv('./data/17_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])   
subreddit_18 = pd.read_csv('./data/18_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])   
subreddit_19 = pd.read_csv('./data/19_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1])   

# Now, we concatenate all the subreddit texts together to get our corpus in one nice dataframe, and save it as a csv
corpus = pd.concat([subreddit_0,subreddit_1,subreddit_2,subreddit_3,subreddit_4,subreddit_5,subreddit_6,subreddit_7,subreddit_8,subreddit_9,subreddit_10,subreddit_11,subreddit_12,subreddit_13,subreddit_14,subreddit_15,subreddit_16,subreddit_17,subreddit_18,subreddit_19], axis=0).reset_index(drop=True)
corpus.to_csv('./data/word_embedding/combined_text.csv')

# The following just strips the corpus of all punctuation and non-alphanumeric characters, and turns everything lower case
# Note, this takes some time to run
for i in range(len(corpus)):
    corpus.text[i] = re.sub(r'\W+', ' ', corpus.text[i]).lower()
corpus.to_csv('./data/word_embedding/processed_combined_text.csv')

# Now, we need to build our word embeddings. We build a frequency table of all the words in the corpus
# and discard words numbering below 50. The rest, we assign a unique index.

# Begin by splitting the sentence by words in the entire corpus dataframe
# Note, if you haven't already, ensure you have run nltk.download('punkt') 
tokenized_corpus = corpus['text'].apply(word_tokenize)
tokenized_corpus.to_csv('./data/word_embedding/tokenized_text.csv')

# We build a frequency table of all the words in corpus. word_freq is of type pd.Series
# Note: what's interesting is that the index is the word, and the attribute is it's frequency
# we can use this to now go and remove words that occur less than 50 times in the corpus
# To access the index use word_freq.index 
word_freq = pd.Series(np.concatenate([x.split() for x in corpus.text])).value_counts()

# vocab is a list of all our words. It'll be our key, and unique_int will be
# our value. We'll make a key-value mapping with these
# vocab[6842] is the first word that has only 19  occurences. Since there are 
# 44,804 words, that means at and after the 6481 st word, all the rest of the words occur 19 times or less

vocab = word_freq.index
unique_int = [0]*len(vocab)
for i in range(len(vocab)):
    if i <= 6841:
        unique_int[i] += i
    else:
        unique_int[i] += -1

# Now we can form our dictionary
dictionary = dict(zip(vocab, unique_int))

# We needed the dictionary so that we can replace words with numbers in the corpus
for lst in tokenized_corpus:
    for ind, item in enumerate(lst):
        lst[ind] = dictionary.get(item, item)

tokenized_corpus.to_csv('./data/word_embedding/integer_tokenized_text.csv')

# Now, we need to add in the label for each sentence, which is the file it came from
list0 = ['0_raw_data']*5000
list1 = ['1_raw_data']*5000
list2 = ['2_raw_data']*5000
list3 = ['3_raw_data']*5000
list4 = ['4_raw_data']*5000
list5 = ['5_raw_data']*5000
list6 = ['6_raw_data']*5000
list7 = ['7_raw_data']*5000
list8 = ['8_raw_data']*5000
list9 = ['9_raw_data']*5000
list10 = ['10_raw_data']*5000
list11 = ['11_raw_data']*5000
list12 = ['12_raw_data']*5000
list13 = ['13_raw_data']*5000
list14 = ['14_raw_data']*5000
list15 = ['15_raw_data']*5000
list16 = ['16_raw_data']*5000
list17 = ['17_raw_data']*5000
list18 = ['18_raw_data']*5000
list19 = ['19_raw_data']*5000

total_list = list0 + list1 + list2 + list3 + list4 + list5 + list6 + list7 + list8 + list9 + list10 + list11 + list12 + list13 + list14 + list15 + list16 + list17 + list18 + list19

df = pd.DataFrame(total_list, columns=['label'])
processed_with_label = pd.concat([tokenized_corpus, df], axis=1)
processed_with_label.to_csv('./data/word_embedding/processed_tokenized_w_label.csv')

# As a final step, we need to create the test and train sets for these embeddings
# and as well, need to seperate the sentences into a numpy array and labels into another numpy array
processed_with_label = processed_with_label.sample(frac=1, random_state=42).reset_index(drop=True)

train_set = processed_with_label[0:70000]
test_set = processed_with_label[70000:100000]

train_set.to_csv('./data/word_embedding/train.csv')
test_set.to_csv('./data/word_embedding/test.csv')

train_X = train_set['text'].to_numpy()
train_y = train_set['label'].to_numpy()

test_X = test_set['text'].to_numpy()
test_y = test_set['label'].to_numpy()

np.save('./data/word_embedding/train_X.npy',train_X)
np.save('./data/word_embedding/train_y.npy',train_y)
np.save('./data/word_embedding/test_X.npy',test_X)
np.save('./data/word_embedding/test_y.npy',test_y)

