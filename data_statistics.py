import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re

word_freq = pd.read_csv('dict.csv', header=None, names=['frequency'])
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')

corpus = pd.concat([test,train,valid], axis=0)
corpus.drop(corpus.columns[0], axis=1, inplace=True)

number_of_words = word_freq['frequency'].sum() 
number_of_unique_words = len(word_freq) 

vocab = word_freq.index

num_common_words = len(word_freq[word_freq['frequency']>=100]) # Words that occur more than or equal to 100 times
num_uncommon_words = len(word_freq[word_freq['frequency']<100]) # Words that occur less than 100 times

num_of_common_words_in_corpus = word_freq[word_freq['frequency']>=100].sum()
num_of_uncommon_words_in_corpus = word_freq[word_freq['frequency']<100].sum()

plt.pie([num_common_words, num_uncommon_words], labels=['common words', 'uncommon words'], autopct='%.1f%%')

plt.pie([num_of_common_words_in_corpus, num_of_uncommon_words_in_corpus], labels=['common words', 'uncommon words'], autopct='%.1f%%')

top_100_words_corpus = word_freq.index[0:100]

for m in range(0,20):
    subreddit0 = corpus[corpus['label'] == m]
    word_freq0 = pd.Series(np.concatenate([x.split() for x in subreddit0.text])).value_counts()
    top_100_words = []
    for i in range(1,101):
        top_100_words.append(vocab[[int(s) for s in re.findall(r'\b\d+\b', word_freq0.index[i])][0]])
    
    similiarity = len(set(top_100_words_corpus).intersection(top_100_words))
    print(similiarity)
    