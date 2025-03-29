import os
import glob
import numpy as np
import torch
import cv2

from torch.utils.data import Dataset


def get_loader(config):
    train_ds = Customdataset(config.data.data_root, 'train')
    valid_ds = Customdataset(config.data.data_root, 'test')

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=config.eval.batch_size, shuffle=False, num_workers=0, drop_last=False)
    
    return train_dataloader, valid_dataloader


class Customdataset(Dataset):
    def __init__(self, data_root, mode):
        self.T2_data_path = glob.glob(os.path.join(data_root, mode, '*/T2/*.npy'))
        self.T2_data_path = sorted(self.T2_data_path)

    def __len__(self):
        return len(self.T2_data_path)
        
    def __getitem__(self, index):
        # T2 dataset
        T2_path = self.T2_data_path[index]
        T2_array = np.load(T2_path).astype(np.float32)
        if T2_array.shape != (256, 256):
            T2_array = cv2.resize(T2_array, (256, 256))
        T2_array = T2_array[np.newaxis, :, :]
        T2_array = T2_array / 255.0
        T2_array = 2 * T2_array - 1

        # T1CE dataset
        T1_path = T2_path.replace('/T2/', '/T1CE/')
        T1_array = np.load(T1_path).astype(np.float32)
        if T1_array.shape != (256, 256):
            T1_array = cv2.resize(T1_array, (256, 256))
        T1_array = T1_array[np.newaxis, :, :]
        T1_array = T1_array / 255.0
        T1_array = 2 * T1_array - 1

        # FLAIR dataset
        FLAIR_path = T2_path.replace('/T2/', '/FLAIR/')
        FLAIR_array = np.load(FLAIR_path).astype(np.float32)
        if FLAIR_array.shape != (256, 256):
            FLAIR_array = cv2.resize(FLAIR_array, (256, 256))
        FLAIR_array = FLAIR_array[np.newaxis, :, :]
        FLAIR_array = FLAIR_array / 255.0
        FLAIR_array = 2 * FLAIR_array - 1

        input = np.concatenate([T1_array, T2_array, FLAIR_array], 0)

        return input
