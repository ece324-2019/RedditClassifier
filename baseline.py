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
        output = self.softmax(output)

        return output

# auto-generate embedding vectors -->
# bag of words