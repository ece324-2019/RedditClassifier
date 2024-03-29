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
        
        self.fc1 = nn.Linear(840, hidden_layer)
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

class CE_CNN(nn.Module):
    def __init__(self, dim_embedding, num_classes, n_filters):
        super(CE_CNN, self).__init__()
        hidden_layer = 1024
        self.conv1 = nn.Conv1d(dim_embedding, n_filters[0], (3), stride=1).float()
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], (3), stride=1).float()
        self.fc1 = nn.Linear(2880, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_classes)
        self.maxpool = torch.nn.MaxPool1d(3, stride=2)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths=None):
        #x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        x1 = self.maxpool(x1)

        x1 = torch.reshape(x1, (x1.shape[0], -1))
        x1 = self.dropout(x1)
        x = self.fc1(x1)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        x = x.squeeze()
        return x

class CE_CNN_Deep(nn.Module):
    def __init__(self, dim_embedding, num_classes, n_filters):
        super(CE_CNN_Deep, self).__init__()
        hidden_layer = 1024
        self.conv1 = nn.Conv1d(dim_embedding, n_filters[0], (3), stride=1).float()
        self.conv2 = nn.Conv1d(n_filters[0], n_filters[1], (3), stride=1).float()
        self.conv3 = nn.Conv1d(n_filters[1], n_filters[2], (3), stride=1).float()
        self.conv4 = nn.Conv1d(n_filters[2], n_filters[3], (3), stride=1).float()
        self.fc1 = nn.Linear(1536, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, num_classes)
        self.maxpool = torch.nn.MaxPool1d(3, stride=2)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, lengths=None):
        #x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x1 = self.conv1(x)
        x1 = F.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.conv3(x1)
        x1 = F.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.conv4(x1)
        x1 = F.relu(x1)
        #x1 = self.maxpool(x1)


        x1 = torch.reshape(x1, (x1.shape[0], -1))
        x1 = self.dropout(x1)
        x = self.fc1(x1)
        x = F.relu(x)
        x = self.fc2(x)
        #x = self.softmax(x)
        x = x.squeeze()
        return x

class CE_CNN_Block(nn.Module):

    def __init__(self, dim_embedding, num_classes, n_filters):
        super(CE_CNN_Block, self).__init__()
        self.block1 = nn.ModuleList(self.block(dim_embedding, n_filters[0]))
        self.block2 = nn.ModuleList(self.block(n_filters[0], n_filters[1]))
        self.block3 = nn.ModuleList(self.block(n_filters[1], n_filters[2]))
        self.block4 = nn.ModuleList(self.block(n_filters[2], n_filters[3]))

        hidden_layer = 1024
        self.fc1 = nn.Linear(n_filters[3], hidden_layer) # n_filters[3]
        self.fc2 = nn.Linear(hidden_layer, num_classes)
        self.maxpool = torch.nn.MaxPool1d(3, stride=2)
        self.maxpool_last = torch.nn.MaxPool1d(21)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def block(self, input, output):
        c1 = nn.Conv1d(input, output, (3), stride=1)
        b1 = nn.BatchNorm1d(output)
        c2 = nn.Conv1d(output, output, (3), stride=1)
        b2 = nn.BatchNorm1d(output)
        c3 = nn.Conv1d(output, output, (3), stride=1)
        b3 = nn.BatchNorm1d(output)
        c4 = nn.Conv1d(output, output, (3), stride=1)
        b4 = nn.BatchNorm1d(output)
        return [c1, c2, c3, c4, b1, b2, b3, b4]

    def apply_block(self, block, x):
        c1,c2,c3,c4,b1,b2,b3,b4 = block
        x = F.relu(b1(c1(x)))
        x = F.relu(b2(c2(x)))
        x = F.relu(b3(c3(x)))
        x = F.relu(b4(c4(x)))
        return x


    def forward(self, x, lengths=None):
        #x = self.dropout(x)
        x = x.permute(0, 2, 1)

        x = self.apply_block(self.block1, x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.apply_block(self.block2, x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.apply_block(self.block3, x)
        x = self.maxpool(x)

        x = self.apply_block(self.block4, x)
        #print(x.shape)
        x = self.maxpool_last(x)
        #x = self.maxpool(x)
        #print(x.shape)

        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = x.squeeze()
        return x

class CE_ResNet(nn.Module):

    def __init__(self, dim_embedding, num_classes, n_filters):
        super(CE_ResNet, self).__init__()

        self.first_conv = nn.Conv1d(dim_embedding, n_filters[0], (3), stride=1, padding=1)

        self.block1 = nn.ModuleList(self.block(n_filters[0], n_filters[0]))
        self.block2 = nn.ModuleList(self.block(n_filters[0], n_filters[0]))

        self.block3 = nn.ModuleList(self.block(n_filters[0], n_filters[1]))
        self.block4 = nn.ModuleList(self.block(n_filters[1], n_filters[1]))

        self.block5 = nn.ModuleList(self.block(n_filters[1], n_filters[2]))
        self.block6 = nn.ModuleList(self.block(n_filters[2], n_filters[2]))

        self.block7 = nn.ModuleList(self.block(n_filters[2], n_filters[3]))
        self.block8 = nn.ModuleList(self.block(n_filters[3], n_filters[3]))

        self.res1 = torch.nn.Conv1d(n_filters[0], n_filters[1], (1), stride=2, padding=0)
        self.res2 = torch.nn.Conv1d(n_filters[1], n_filters[2], (1), stride=2, padding=0)
        self.res3 = torch.nn.Conv1d(n_filters[2], n_filters[3], (1), stride=2, padding=0)
        self.bn1 = nn.BatchNorm1d(n_filters[1])
        self.bn2 = nn.BatchNorm1d(n_filters[2])
        self.bn3 = nn.BatchNorm1d(n_filters[3])

        hidden_layer = 1024
        self.fc1 = nn.Linear(n_filters[3], hidden_layer) # n_filters[3]
        self.fc2 = nn.Linear(hidden_layer, num_classes)
        self.maxpool = torch.nn.MaxPool1d(3, stride=2, padding=1)
        self.maxpool_last = torch.nn.MaxPool1d(36)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def block(self, input, output):
        padding = [1,1]
        p1 = nn.ConstantPad1d(padding, 0)
        c1 = nn.Conv1d(input, output, (3), stride=1, padding=0)
        p2 = nn.ConstantPad1d(padding, 0)
        b1 = nn.BatchNorm1d(output)
        c2 = nn.Conv1d(output, output, (3), stride=1, padding=0)
        b2 = nn.BatchNorm1d(output)
        return [c1, c2, b1, b2, p1, p2]

    def apply_block(self, block, x):
        c1,c2,b1,b2,p1,p2 = block
        x = F.relu(b1(c1(p1(x))))
        x = F.relu(b2(c2(p2(x))))
        return x

    def apply_residue_conv(self, x, res, bn):
        x = bn(res(x))
        return x

    def forward(self, x, lengths=None):
        #x = self.dropout(x)
        x = x.permute(0, 2, 1)
        x = self.first_conv(x)
        r1 = x
        x = self.apply_block(self.block1, x)
        r2 = r1 + x
        x = self.apply_block(self.block2, x)
        x = r2 + x
        r3 = x
        x = self.maxpool(x)
        x = self.apply_block(self.block3, x)
        x = self.apply_residue_conv(r3, self.res1, self.bn1) + x
        r4 = x
        x = self.apply_block(self.block4, x)
        x = r4 + x
        r5 = x
        x = self.maxpool(x)
        x = self.apply_block(self.block5, x)
        x = self.apply_residue_conv(r5, self.res2, self.bn2) + x
        r6 = x
        x = self.apply_block(self.block6, x)
        x = r6 + x
        r7 = x
        x = self.maxpool(x)
        x = self.apply_block(self.block7, x)
        x = self.apply_residue_conv(r7, self.res3, self.bn3) + x
        r8 = x
        x = self.apply_block(self.block8, x)
        x = r8 + x
        x = self.maxpool_last(x)

        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        x = x.squeeze()
        return x
