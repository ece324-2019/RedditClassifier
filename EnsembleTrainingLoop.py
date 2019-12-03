import pandas as pd 

import pandas as pd
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchtext
from torchtext import data
import spacy
import en_core_web_sm
import argparse
import os
import math
import matplotlib.pyplot as plt

from EnsembleModel import ComboCNN

X_test = np.load('test_X.npy', allow_pickle=True)
X_train = np.load('train_X.npy', allow_pickle=True)
X_valid = np.load('valid_X.npy', allow_pickle=True)

y_test = np.load('test_y.npy', allow_pickle=True)
y_train = np.load('train_y.npy', allow_pickle=True)
y_valid = np.load('valid_y.npy', allow_pickle=True)

 This is just text that has been stripped of grammar and non-alphanumeric characters, and is all lower case
corpus = pd.read_csv(r'C:\Users\Pranav\Desktop\Reddit Classifier Project Data Processing\data\word_embedding\processed_combined_text.csv')

# Now, we need to add in the label for each sentence, which is the file it came from
list0 = [0]*5000
list1 = [1]*5000
list2 = [2]*5000
list3 = [3]*5000
list4 = [4]*5000
list5 = [5]*5000
list6 = [6]*5000
list7 = [7]*5000
list8 = [8]*5000
list9 = [9]*5000
list10 = [10]*5000
list11 = [11]*5000
list12 = [12]*5000
list13 = [13]*5000
list14 = [14]*5000
list15 = [15]*5000
list16 = [16]*5000
list17 = [17]*5000
list18 = [18]*5000
list19 = [19]*5000

total_list = list0 + list1 + list2 + list3 + list4 + list5 + list6 + list7 + list8 + list9 + list10 + list11 + list12 + list13 + list14 + list15 + list16 + list17 + list18 + list19

df = pd.DataFrame(total_list, columns=['label'])
processed_with_label = pd.concat([corpus, df], axis=1)
processed_with_label.to_csv('processed_corpus_labelled.csv')

# Drop the redundant indexing column
processed_with_label.drop('Unnamed: 0', axis=1, inplace=True)

# Shuffle the data around
processed_with_label = processed_with_label.sample(frac=1, random_state=42).reset_index(drop=True)

train_set = processed_with_label[0:70000]
valid_set = processed_with_label[70000:85000]
test_set = processed_with_label[85000:100000]

train_set.to_csv('my_train.csv')
valid_set.to_csv('my_valid.csv')
test_set.to_csv('my_test.csv')



  
def main(args):


    TEXT = data.Field(sequential=True,lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)
    
    # 3.2.2
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='./data/', train='my_train.csv',
            validation='my_valid.csv', test='my_test.csv', format='csv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])
    
    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
      (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
    	sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    
    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data)
    
    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab
    # Initialize the baseline_model, optimizer, loss function, and some auxiliary 
    # functions for plotting accuracy and loss 
    
    choose_model = args.model

    if choose_model == 'combocnn':
        model = ComboCNN(args.emb_dim, TEXT.vocab, args.num_filt, [2,3])
       
 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fnc = nn.CrossEntropyLoss(reduction='mean')
    
    loss_array = [0]*args.epochs
    acc_array = [0]*args.epochs
    
    valid_loss = [0]*args.epochs
    valid_acc = [0]*args.epochs

    
    epoch_array = [0]*args.epochs
    for i in range(args.epochs):
        epoch_array[i]= i
    
    loss_train = 0
    correct_train = 0
    
    loss_valid = 0
    correct_valid = 0
    

    
    

    
    def accuracy(prediction, label): # Need a way to measure accuracy. We do this by taking the max index of the output tensor
        num_correct = 0
        
        pred = prediction.clone().detach().numpy() # We need to copy the values of these tensors since they have autograd and we dont wanna mess that up by detaching that or returning a processed version
        lab = label.clone().detach().numpy() # So, we use .clone() to copy, .detach() to remove from the autograd computation graph, and .numpy() to make manipulations easier
    
        for i in range(len(pred)):
            if np.argmax(pred[i]) == np.argmax(lab[i]): # Each element of prediction and label is a length 10 vector. In each vector, the index of the max element refers to the class that's most strongly predicted.
                num_correct += 1                                # If the indices where they have their max elements line up, it means the predicted strongest class is the same as the actual class. Hence, it's a correct prediction.
        #acc = float(num_correct/len(pred))
        return num_correct
    
    def turn1hot(tensor_array): # A function to one-hot encode the labels of each batch 
        nparr = tensor_array.clone().detach().numpy() # turn the tensor into a numpy array after detaching it from any computational graphs
        arr = nparr.tolist() # Turn the array into a normal list for easy manipulation
        for i in range(len(arr)): # For loop does the process for the whole batch
            val_at_index = nparr[i]
            zero_vec = [0]*20 # Create a list of 0s of length number of classes 
            zero_vec[int(val_at_index)] += 1 # Make the appropriate class's index a 1 
            arr[i] = zero_vec
        return np.asarray(arr) # Return the one hot encoded list as an array

    
    # Training loop for train and validation set 
    for i in range(0,args.epochs,1):
        model.train()
        for batch_index, batch_data in enumerate(train_iter):
            optimizer.zero_grad() # Zero gradients at the very start of each batch
            
            input_data, input_data_length = batch_data.text
            #input_data = input_data.float()
            #input_data_length = input_data_length.float()
            labels = batch_data.label
            
            onehotlabel = turn1hot(labels)
            inputlabels = torch.from_numpy(onehotlabel).long()
            
            if args.model == 'rnn': 
                predictions = model(input_data, input_data_length) # Make a forward pass 
            else:
                predictions = model(input_data)
            loss = loss_fnc(predictions, labels) # Compute the loss real quick ya feel 
            
            loss_train += loss
            correct_train += accuracy(predictions, inputlabels) # Get the accuracy for this epoch by running this through the accuracy function
        
            loss.backward()
            optimizer.step()
        
        model.eval()
        for batch_index, batch_data in enumerate(val_iter):
            
            input_data, input_data_length = batch_data.text
            labels = batch_data.label

            onehotlabel = turn1hot(labels)
            inputlabels = torch.from_numpy(onehotlabel).long()
            
            if args.model == 'rnn': 
                predictions = model(input_data, input_data_length) # Make a forward pass 
            else:
                predictions = model(input_data)
            loss = loss_fnc(predictions, labels) # Compute the loss real quick ya feel 
            
            loss_valid += loss
            correct_valid += accuracy(predictions, inputlabels) # Get the accuracy for this epoch by running this through the accuracy function


        loss_array[i] += float(loss_train/(math.ceil(len(train_data)/args.batch_size))) # loss_train sums the mean of loss per batch. so for the whole epoch, we take the mean loss of the means of each batch. So, loss_train divide by how many batches we had. We have len(overfit_data) amount of data, and args.batch_size size of batch. So divide the two
        acc_array[i] += float(correct_train/len(train_data))
        
        valid_loss[i] += float(loss_valid/(math.ceil(len(val_data)/args.batch_size))) # loss_train sums the mean of loss per batch. so for the whole epoch, we take the mean loss of the means of each batch. So, loss_train divide by how many batches we had. We have len(overfit_data) amount of data, and args.batch_size size of batch. So divide the two
        valid_acc[i] += float(correct_valid/len(val_data))
        
#        
        loss_train = 0
        correct_train = 0
        
        loss_valid = 0
        correct_valid = 0
        

    """PLOTTING DATA"""
    
    font = {'size': 10}
    plt.figure(1)  # Figure 1 is for training and validation loss + accuracy
    plt.subplot(211)
    plt.plot(epoch_array, loss_array, 'r', epoch_array, valid_loss, 'g')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training loss', 'validation loss'])
    plt.subplot(212)
    plt.plot(epoch_array, acc_array, 'r', epoch_array, valid_acc, 'g')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training accuracy', 'validation accuracy'])


    plt.show()
    print("validation acc: " + str(valid_acc[-1]) + " training acc: " + str(acc_array[-1]))
    ######    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='combocnn',
                        help="Model type: baseline,rnn,cnn, combocnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)
