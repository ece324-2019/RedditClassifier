import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as util



class ComboCNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(ComboCNN, self).__init__()
        ######
        # Section 5.0 YOUR CODE HERE
        self.embed = nn.Embedding.from_pretrained(vocab.vectors)
        
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 50, (embedding_dim, 1)),
                        )
        self.conv2 = nn.Conv1d(50, 25, 1)
        self.conv3 = nn.Conv1d(25, 10, 1)
        
        self.bn1 = nn.BatchNorm2d(50)
        self.bn2 = nn.BatchNorm1d(25)
        self.bn3 = nn.BatchNorm1d(10)
        
        self.lin1 = nn.Linear(10,50)
        self.lin2 = nn.Linear(100,50)
        self.lin3 = nn.Linear(100,20)
        ######


    def forward(self, x, lengths=None):
        ######
        # Section 5.0 YOUR CODE HERE
        # This part is for the method of means 
        embedded = self.embed(x)
        average = embedded.mean(0)        
        
        x = self.embed(x)
        # Comes in shape: [26,64,100] = [# words, batch size, embedding dim]
        x = x.permute(1,2,0) # [64, 100, 26]
        x = x.unsqueeze(1) # [64, 1, 100, 26] # 2nd element needs to correspond to number input channels = 1

        x1 = F.relu(self.bn1(self.conv1(x))).squeeze(2)
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
#        x1, wasteman = torch.max(x1,2)
#        x2, wasteman = torch.max(x2,2)
        x3, wasteman = torch.max(x3,2)
        
        #y = torch.cat([x1,x2,x3], 1).squeeze()
        
        cnn_out = self.lin1(x3)
        mean_out = self.lin2(average)
        
        output = self.lin3(torch.cat([cnn_out, mean_out],1).squeeze()) 
        return output
