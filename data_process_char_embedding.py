import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from os import path

import string
import re
import ast
from multiprocessing import Pool

num_threads = 30

# We create a dictionary of all ASCII characters. We'll need this to map the characters to unique integers
dictionary = {chr(i): i for i in range(32,128)}
for key in dictionary:
    dictionary[key] = dictionary[key] - 32

def process_subdata(input_pack):
    corpus, initial = input_pack

    print(len(corpus.text))
    i =0
    while i < len(corpus.text):
        corpus.text[i+initial] = ','.join(str(dictionary[j]) if (j in dictionary) else '-1' for j in
            corpus.text[i+initial].lower())
        print(str(corpus.text[i+initial]) + "\n" + str(list(ast.literal_eval(corpus.text[i+initial]))))
        corpus.text[i+initial] = list(ast.literal_eval(corpus.text[i+initial]))
        if (i % 1000 == 0):
            print(i)
        i += 1
    return corpus

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
# 97 total characters

# The following iterates over each sentence in the corpus, turns all characters lower case, and changes the characters into
# a unique integer. It also turns all non-ascii characters into -1
# Note, this takes some time to run

if not path.exists('data/char_embeddings/checkpoint1.csv'):
    sub_corpus = []
    i = 0
    print(len(corpus))
    section = len(corpus) // num_threads
    while i < num_threads:
        end = (i + 1) * section
        if i == num_threads - 1:
            end = len(corpus)
        sub_corpus.append([corpus[i * section:end], section * i])
        i += 1
    p = Pool(processes=num_threads)
    output = p.map(process_subdata, sub_corpus)
    p.close()
    corpus = pd.concat(output)
    corpus.to_csv('data/char_embeddings/checkpoint1.csv')
else:
    corpus = pd.read_csv('data/char_embeddings/checkpoint1.csv')

# Now, we have to pad all the rows to the same length (of longest sentence)
# and we pad all the shorter sentences with -1 at the end 
'''padded_corpus = pd.DataFrame(corpus['text'].values.tolist()).agg(list, 1)
for i in range(len(padded_corpus)):
    padded_corpus[i] = [-1 if pd.isnull(x) else x for x in padded_corpus[i]]

padded_corpus.to_csv('./data/char_embedding/padded_int_char_encoded_text.csv')
'''

df = pd.DataFrame(subreddit_labels, columns=['label'])
processed_with_label = pd.concat([corpus, df], axis=1)


# As a final step, we need to create the test and train sets for these embeddings
# and as well, need to seperate the sentences into a numpy array and labels into another numpy array
processed_with_label = processed_with_label.sample(frac=1, random_state=42).reset_index(drop=True)

test_ratio = 0.05
valid_ratio = 0.1

test_set = processed_with_label[0:int(len(processed_with_label)*test_ratio)]
valid_set = processed_with_label[int(len(processed_with_label)*test_ratio):int(len(processed_with_label)*(test_ratio+valid_ratio))]
train_set = processed_with_label[int(len(processed_with_label)*(test_ratio+valid_ratio)):]
print(len(test_set))
print(len(train_set))
print(len(valid_set))

train_set.to_csv('data/char_embeddings/train.csv')
valid_set.to_csv('data/char_embeddings/valid.csv')
test_set.to_csv('data/char_embeddings/test.csv')


train_X = train_set['text'].to_numpy()
train_y = train_set['label'].to_numpy()

test_X = test_set['text'].to_numpy()
test_y = test_set['label'].to_numpy()

valid_X = valid_set['text'].to_numpy()
valid_y = valid_set['label'].to_numpy()


np.save('data/char_embeddings/train_X.npy',train_X)
np.save('data/char_embeddings/train_y.npy',train_y)
np.save('data/char_embeddings/test_X.npy',test_X)
np.save('data/char_embeddings/test_y.npy',test_y)
np.save('data/char_embeddings/valid_X.npy',valid_X)
np.save('data/char_embeddings/valid_y.npy',valid_y)


