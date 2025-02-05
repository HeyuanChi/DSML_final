import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, input_length, output_length):
        self.data = data
        self.N = len(self.data)
        self.input_length = input_length
        self.output_length = output_length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.input_length]
        y = self.data[idx + self.input_length -1: idx + self.input_length + self.output_length]

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y) 
        
        return x, y

    def __len__(self):
        return self.N - self.input_length - self.output_length


class BatchSampler():
    def __init__(self, dataset, batch_size):
        self.B = batch_size
        self.dataset = dataset

    def __call__(self):
        batch = [self.dataset[i] for i in self.get_random_inital_conditions()]

        xs = torch.stack([x for x, _ in batch])
        ys = torch.stack([y for _, y in batch])

        return xs, ys
        
    def get_random_inital_conditions(self):
        return torch.randperm(len(self.dataset))[:self.B]