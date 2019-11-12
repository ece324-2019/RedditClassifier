import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):

    def __init__(self, num_words, dim_embedding, num_classes):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding(num_words, dim_embedding)
        self.fc = nn.Linear(dim_embedding, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)
        average = embedded.mean(1) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)
        #output = self.softmax(output)

        return output

class Bag_of_Words(nn.Module):

    def __init__(self, num_words, num_classes):
        super(Bag_of_Words, self).__init__()

        self.embedding = nn.Embedding(num_words, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)
        average = embedded.mean(1) # [sentence length, batch size, embedding_dim]
        output = average
        #output = self.softmax(output)

        return output

class Baseline(nn.Module):

    def __init__(self, num_words, dim_embedding, num_classes):
        super(Baseline, self).__init__()

        self.embedding = nn.Embedding(num_words, dim_embedding)
        self.fc = nn.Linear(dim_embedding, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths=None):
        #x = [sentence length, batch size]
        embedded = self.embedding(x)
        average = embedded.mean(1) # [sentence length, batch size, embedding_dim]
        output = self.fc(average).squeeze(1)
        #output = self.softmax(output)

        return output

class CNN(nn.Module):
    def __init__(self,  num_words, dim_embedding, num_classes, n_filters):
        super(CNN, self).__init__()
        hidden_layer = 1024
        self.embedding = nn.Embedding(num_words, dim_embedding)
        self.conv1 = nn.Conv1d(dim_embedding, n_filters[0], (3), stride=1).float()
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], (3), stride=1).float()
        self.fc1 = nn.Linear(840, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_classes)
        self.maxpool = torch.nn.MaxPool1d(3, stride=2)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        #x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        x1 = torch.reshape(x1, (x1.shape[0], -1))
        x1 = self.dropout(x1)
        x = self.fc1(x1)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        x = x.squeeze()
        return x

class LSTM(nn.Module):
    def __init__(self, num_words, dim_embedding, num_classes, memory_size):
        super(LSTM, self).__init__()
        hidden_size = 1024
        self.embedding = nn.Embedding(num_words, dim_embedding)

        self.lstm = nn.LSTM(dim_embedding, memory_size, batch_first=True)
        self.fc1 = nn.Linear(memory_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        #x = nn.utils.rnn.pack_padded_sequence(x, lengths)
        #x = x.permute(1, 0 ,2)
        output, (hn, cn) = self.lstm(x)
        x = hn.squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        #print(w.shape)
        x = self.fc2(x)
        x = x.squeeze()
        return x

class LSTM_Deep(nn.Module):
    def __init__(self, num_words, dim_embedding, num_classes, memory_size):
        super(LSTM_Deep, self).__init__()
        hidden_size = 1024
        self.embedding = nn.Embedding(num_words, dim_embedding)

        self.lstm1 = nn.LSTM(dim_embedding, memory_size, batch_first=True)
        self.lstm2 = nn.LSTM(memory_size, memory_size*2, batch_first=True)
        self.lstm3 = nn.LSTM(memory_size*2, memory_size*2, batch_first=True)

        self.fc1 = nn.Linear(memory_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def forward(self, x, lengths=None):
        x = self.embedding(x)

        output, (hn, cn) = self.lstm1(x)
        #output = self.dropout(F.relu(output))

        output, (hn, cn) = self.lstm2(output)
        #output = self.dropout(F.relu(output))

        output, (hn, cn) = self.lstm3(output)
        # hn = self.dropout(F.relu(hn))
        x = hn.squeeze()
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.squeeze()
        return x

class CNN_Deep(nn.Module):
    def __init__(self, num_words, dim_embedding, num_classes, n_filters):
        super(CNN_Deep, self).__init__()
        hidden_layer = 1024
        self.embedding = nn.Embedding(num_words, dim_embedding)
        self.conv1 = nn.Conv1d(dim_embedding, n_filters[0], (3), stride=1).float()
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], (3), stride=1).float()
        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], (3), stride=1).float()
        self.conv4 = nn.Conv1d(n_filters[2], n_filters[3], (3), stride=1).float()
        
        self.fc1 = nn.Linear(864, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_classes)
        self.maxpool = torch.nn.MaxPool1d(3, stride=2)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.maxpool(x)
        x = self.dropout(F.relu(x))
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = torch.reshape(x, (x.shape[0], -1))
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        x = x.squeeze()
        return x
