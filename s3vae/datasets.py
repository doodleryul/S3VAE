import numpy as np
import cv2
import os
import ast
import pandas as pd
from torch.utils.data import Dataset


class SeqMNISTDataset(Dataset):
    def __init__(self, filename):
        self.seq_mnist = np.load(filename)

    def __len__(self):
        return self.seq_mnist.shape[1]  # T,N,W,H

    def __getitem__(self, idx):
        outputs = np.expand_dims(self.seq_mnist, axis=2)
        return outputs[:,idx,:,:,:]

class FaceSeqDataset(Dataset):
    def __init__(self, root_dir, csv_filename, transform=None):
        self.image_channel = 3
        self.image_shape =(64, 64)
        self.root_dir = root_dir
        self.image_frame = pd.read_csv(csv_filename)

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):
        image_names, labels = map(ast.literal_eval, self.image_frame.iloc[idx])

        outputs = np.empty((0, self.image_channel, *self.image_shape), int)
        for image_name in image_names:
            image = cv2.imread(os.path.join(self.root_dir, image_name))
            image = cv2.resize(image, self.image_shape)
            image = np.transpose(image, (2,0,1))
            image = image.reshape(1, *image.shape)
            outputs = np.append(outputs, image, axis=0)

        return outputs, np.array(labels)
