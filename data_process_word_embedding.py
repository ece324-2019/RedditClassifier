"""
Code to process text data, build the word embeddings, and ready it for NLP pipeline 
"""
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from os import path

import string
import re

number_subreddits = 20
subreddit_labels = []
# First load the data in, and we only need the text from the files, so we load in column 1, skip the first header row and use the custom header of (text) as it's just a bit more descriptive
subreddits = []
i =0
while i < number_subreddits:
    subreddits.append(pd.read_csv('data/' + str(i) + '_raw_data.csv', header=None, names=['text'], skiprows=[0], usecols=[1], lineterminator='\n'))
    j = 0
    while j < len(subreddits[i]):
        subreddit_labels.append(i)
        j += 1
    i += 1

# Now, we concatenate all the subreddit texts together to get our corpus in one nice dataframe, and save it as a csv
corpus = pd.concat(subreddits, axis=0).reset_index(drop=True)
#corpus.to_csv('data/word_embedding/combined_text.csv')

# The following just strips the corpus of all punctuation and non-alphanumeric characters, and turns everything lower case
# Note, this takes some time to run
if not path.exists('data/word_embeddings/checkpoint1.csv'):
    for i in range(len(corpus)):
        corpus.text[i] = re.sub(r'\W+', ' ', corpus.text[i]).lower()
        if i % 100 == 0:
            print(i)
    corpus.to_csv('data/word_embeddings/checkpoint1.csv')
else:
    corpus = pd.read_csv('data/word_embeddings/checkpoint1.csv')

# Now, we need to build our word embeddings. We build a frequency table of all the words in the corpus
# and discard words numbering below 50. The rest, we assign a unique index.

# Begin by splitting the sentence by words in the entire corpus dataframe
# Note, if you haven't already, ensure you have run nltk.download('punkt') 
tokenized_corpus = corpus['text'].apply(word_tokenize)
#tokenized_corpus.to_csv('data/word_embedding/tokenized_text.csv')

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
unique_int = []
threshold = 20
for i in range(len(vocab)):
    if word_freq[i] >= threshold:
        unique_int.append(i)
    else:
        unique_int.append(-1)


# Now we can form our dictionary
dictionary = dict(zip(vocab, unique_int))

# We needed the dictionary so that we can replace words with numbers in the corpus
for lst in tokenized_corpus:
    for ind, item in enumerate(lst):
        lst[ind] = dictionary.get(item, item)

#tokenized_corpus.to_csv('./data/word_embedding/integer_tokenized_text.csv')

df = pd.DataFrame(subreddit_labels, columns=['label'])
processed_with_label = pd.concat([tokenized_corpus, df], axis=1)
processed_with_label.to_csv('data/word_embeddings/checkpoint2.csv')

# As a final step, we need to create the test and train sets for these embeddings
# and as well, need to seperate the sentences into a numpy array and labels into another numpy array
processed_with_label = processed_with_label.sample(frac=1, random_state=42).reset_index(drop=True)
test_ratio = 0.1
valid_ratio = 0.2

test_set = processed_with_label[0:int(len(processed_with_label)*test_ratio)]
valid_set = processed_with_label[int(len(processed_with_label)*test_ratio):int(len(processed_with_label)*(test_ratio+valid_ratio))]
train_set = processed_with_label[int(len(processed_with_label)*(test_ratio+valid_ratio)):]
print(len(test_set))
print(len(train_set))
print(len(valid_set))

train_set.to_csv('data/word_embeddings/train.csv')
valid_set.to_csv('data/word_embeddings/valid.csv')
test_set.to_csv('data/word_embeddings/test.csv')


train_X = train_set['text'].to_numpy()
train_y = train_set['label'].to_numpy()

test_X = test_set['text'].to_numpy()
test_y = test_set['label'].to_numpy()

valid_X = valid_set['text'].to_numpy()
valid_y = valid_set['label'].to_numpy()


np.save('data/word_embeddings/train_X.npy',train_X)
np.save('data/word_embeddings/train_y.npy',train_y)
np.save('data/word_embeddings/test_X.npy',test_X)
np.save('data/word_embeddings/test_y.npy',test_y)
np.save('data/word_embeddings/valid_X.npy',valid_X)
np.save('data/word_embeddings/valid_y.npy',valid_y)

