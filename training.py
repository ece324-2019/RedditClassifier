import torchtext
from torchtext import data
import torch

batch_size = 64

def load_data():



    train_data, val_data, test_data = data.TabularDataset.splits(
        path='data/', train='train.csv',
        validation='validation.tsv', test='test.tsv', format='tsv',
        skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(batch_size, batch_size, batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)