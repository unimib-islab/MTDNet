import os
import random

import ipdb
import numpy as np
import pandas as pd
import scipy
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Dataset

__all__ = ["SimpleGraphDataset", "CWTGraphDataset", "CWTPyGDataGraphDataset"]


class SimpleDataset(Dataset):
    def __init__(self, annot_df, dataset_crop_path, mult_noise=False, add_noise=False, mode='train'):
        super().__init__()
        self.annot_df = annot_df
        self.dataset_crop_path = os.path.join(dataset_crop_path + '/data')
        self.mult_noise = mult_noise
        self.add_noise = add_noise
        self.mode = mode
        if "APAVA" in dataset_crop_path:
            self.n_el=16
        elif "brainlat" in dataset_crop_path:
            self.n_el=128
        else:
            self.n_el=19

    def len(self):
        return len(self.annot_df)
    
    def get_electrode_number(self):
        return self.n_el

    def augmentation(self, x):

        # random time flip
        if random.uniform(0, 1) > 0.5:
            x = np.flip(x, 1)
        # random electrode shuffling
        if random.uniform(0, 1) > 0.5:
            np.random.shuffle(x) #shuffle the first dimension
        # # random value masking
        if random.uniform(0, 1) > 0.5:
            indices = np.random.choice(np.arange(x.shape[1]), replace=False,
                           size=int(x.shape[1] * 0.2))
            x[:,indices] = 0
        
        if x is None:
            import ipdb; ipdb.set_trace()
        return x


    def get(self, idx):

        record = self.annot_df.iloc[idx]
        file_path = os.path.join(self.dataset_crop_path, record['crop_file'])
        crop = np.load(file_path)


        n_el, n_samples = crop.shape

        # center in zero
        scaler = StandardScaler(with_std=False)
        scaler.fit(crop.transpose(1,0))
        crop = scaler.transform(crop.transpose(1,0)).transpose(1,0)
        
        # normalize
        scaler2 = StandardScaler()
        scaler2.fit(crop.reshape(n_el*n_samples,1))
        crop = scaler2.transform(crop.reshape(n_el*n_samples,1)).reshape(n_el,n_samples)

        freq_signal = crop
        if self.mode == 'train':
            freq_signal = self.augmentation(freq_signal)
        freq_signal = torch.tensor(
            freq_signal.astype(np.float32))

        freq_signal = freq_signal.contiguous()
        

        return freq_signal, torch.tensor([record['label']]), record['original_rec']
