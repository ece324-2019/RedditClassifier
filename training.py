
import torch
import torch.optim as optim

import numpy as np
from models import Baseline, Bag_of_Words, CNN, CNN_Deep, RNN
import matplotlib.pyplot as plt

batch_size = 64
target_length = 50 # 50
num_epochs = 50
learning_rate = 0.001
num_words, dim_embedding = 11400, 100 # 100
num_classes = 20

base_path = "data/"
word_path = "word_embeddings/" # char_embeddings

# load data, create batches

def load_data(x_path, y_path, target_length):

    # load data
    X = np.load(x_path,allow_pickle=True)
    y = np.load(y_path,allow_pickle=True)

    # gets max possible length
    '''max_length = 0
    i =0
    while i < len(X):
        max_length = max(len(X[i]), max_length)
        i += 1
    print(max_length)'''
    # gives percent entailed by a certain length for truncation
    '''
    count = 0
    i = 0
    while i < len(X):
        if len(X[i]) <= target_length:
            count += 1
        i += 1
    count = float(count) / len(X)
    print(count)
    '''
    # pad data, remove some strange string character called lem/gim that made their way to input data

    padding = np.full((target_length), -1)
    i = 0
    while i < len(X):

        if len(X[i]) < target_length:
            try:
                X[i] = np.concatenate([np.array(X[i], dtype=int),padding[0:target_length-len(X[i])]], 0)
            except:
                j = 0
                while j < len(X[i]):
                    try:
                        int(X[i][j])
                    except:
                        X[i][j] = -1
                    j += 1
                X[i] = np.concatenate([np.array(X[i], dtype=int), padding[0:target_length - len(X[i])]], 0)
        else:
            try:
                X[i] = np.array((X[i][0:target_length]), dtype=int)
            except:
                j = 0
                while j < len(X[i]):
                    try:
                        int(X[i][j])
                    except:
                        X[i][j] = -1
                    j += 1
                X[i] = np.array((X[i][0:target_length]), dtype=int)
            #print(len(X[i]))
        i += 1
        #print(len(X[i]))

    X = np.array(list(X), dtype=int)
    X = X + 1


    batched_X = np.array_split(X, len(X)//batch_size + 1)

    batched_y = np.array_split(y, len(y) // batch_size + 1)

    # lines up batches by adding several repetitions to the last batch. In this dataset's case it is only 1 sample
    if len(X) % batch_size != 0:
        gap = batch_size - len(batched_X[-1])
        to_concat = X[0:gap]
        #print(gap)
        #print(len(batched_X[-1]))
        batched_X[-1] = np.concatenate([batched_X[-1], to_concat])
        #print(len(batched_X[-1]))

        to_concat = y[0:gap]
        batched_y[-1] = np.concatenate([batched_y[-1], to_concat])

    batched_X = np.array(batched_X)
    batched_y = np.array(batched_y)
    print(len(batched_X))
    #print(len(batched_X[0]))
    #print(len(batched_X[0][0]))

    return batched_X, batched_y

def plot_tri(a, title):
    a = np.array(a)

    plt.plot(a[:, 0], a[:, 2], label="Train Accuracy")
    plt.plot(a[:, 0], a[:, 4], label="Valid Accuracy")
    plt.plot(a[:, 0], a[:, 6], label="Test Accuracy")
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch count')
    t = "Accuracy for " + title
    plt.title(t)
    plt.savefig("figs/" + t + ".png")
    plt.clf()

    plt.plot(a[:, 0], a[:, 1], label="Train Loss")
    plt.plot(a[:, 0], a[:, 3], label="Valid Loss")
    plt.plot(a[:, 0], a[:, 5], label="Test Loss")

    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch count')
    t = "Loss for " + title
    plt.title(t)
    plt.savefig("figs/" + t + ".png")
    plt.clf()

def train_model(data_pack, num_epochs, learning_rate, num_words, dim_embedding, num_classes):
    train_X, train_y, valid_X, valid_y, test_X, test_y = data_pack

    model_name = "Shallow-CNN" # Shallow-RNN, Baseline-AvEmbedding, Baseline-BoW
    if model_name == "Baseline-BoW":
        model = Bag_of_Words(num_words, num_classes)
    elif model_name == "Baseline-AvEmbedding":
        model = Baseline(num_words, dim_embedding, num_classes)
    elif model_name == "Shallow-CNN":
        n_filters = [40, 40]
        model = CNN(num_words, dim_embedding, num_classes, n_filters)
    elif model_name == "Shallow-RNN":
        memory_size = 100
        model = RNN(num_words, dim_embedding, num_classes, memory_size)

    #n_filters = [15, 20, 40]
    #model = CNN_Deep(num_words, dim_embedding, num_classes, n_filters)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    reduce_size = 0.2
    a = []

    epoch = 0
    while epoch < num_epochs:
        i =0
        s1 = np.random.choice(range(len(train_X)), int(reduce_size*len(train_X)), replace=False)
        while i < len(s1):
            # for each batch......... ????
            optimizer.zero_grad()
            batch_x = train_X[s1[i]]
            batch_y = train_y[s1[i]]
            batch_x = torch.Tensor(batch_x).type('torch.LongTensor')
            output = model(batch_x)
            batch_y = torch.Tensor(batch_y).type('torch.LongTensor')

            loss = criterion(output, batch_y)
            #print(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            i += 1
        model.eval()
        t_loss, t_acc = run_testing(model, criterion, train_X[s1], train_y[s1])
        v_loss, v_acc = run_testing(model, criterion, valid_X, valid_y)
        tt_loss, tt_acc = run_testing(model, criterion, test_X, test_y)
        a.append([i, t_loss, t_acc, v_loss, v_acc, tt_loss, tt_acc])
        model.train()
        #print(t_loss)
        print(str(t_acc) + " " + str(v_acc))
        epoch += 1
    plot_tri(a, model_name)

def run_testing(model, criterion, train_X, train_y):
    t_loss, t_acc, t_sum = 0, 0, 0
    i = 0
    while i < len(train_X):

        batch_x = train_X[i]
        batch_y = train_y[i]
        batch_x = torch.Tensor(batch_x).type('torch.LongTensor')
        batch_y = torch.Tensor(batch_y).type('torch.LongTensor')

        output = model(batch_x)
        loss = criterion(output, batch_y)
        #print(output.shape)
        accuracy = torch.argmax(output, 1)

        #print(batch_y)
        #print(accuracy)

        accuracy = (torch.sum(torch.eq(accuracy, batch_y).type('torch.LongTensor'), dim=0) / float(batch_y.shape[0])).detach().numpy()
        #accuracy = torch.gt(output, 0.5).type('torch.LongTensor')
        #accuracy = torch.eq(accuracy, target).type('torch.DoubleTensor')
        #accuracy = (torch.sum(accuracy, dim=0) / data.shape[1]).detach().numpy()
        t_sum += 1
        t_acc += accuracy
        t_loss += loss.detach().numpy()
        i += 1
    t_acc = t_acc / t_sum
    t_loss = t_loss / t_sum
    return t_loss, t_acc

train_X, train_y = load_data(base_path + word_path + "train_X.npy", base_path + word_path + "train_y.npy", target_length)
valid_X, valid_y = load_data(base_path + word_path + "valid_X.npy", base_path + word_path + "/valid_y.npy", target_length)
test_X, test_y = load_data(base_path + word_path + "test_X.npy", base_path + word_path + "/test_y.npy", target_length)


train_model([train_X, train_y, valid_X, valid_y, test_X, test_y], num_epochs, learning_rate, num_words, dim_embedding, num_classes)

