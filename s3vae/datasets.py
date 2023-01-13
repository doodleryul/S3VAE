import numpy as np
from torch.utils.data import Dataset


class SeqMNISTDataset(Dataset):
    def __init__(self, filename):
        self.seq_mnist = np.load(filename)

    def __len__(self):
        return self.seq_mnist.shape[1]  # T,N,W,H

    def __getitem__(self, idx):
        outputs = np.expand_dims(self.seq_mnist, axis=2)
        return outputs[:,idx,:,:,:]