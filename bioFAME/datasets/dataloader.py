import torch
from torch.utils.data import Dataset

import os
import numpy as np

import logging


logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger('bioFAME')
logging.getLogger('matplotlib.font_manager').disabled = True


# dataset channel to index
SleepEDF_multi = {
    'training-set-fixed-scaling': 1,

    # the commonly used single EEG channel
    'EEG Fpz-Cz': 0, 
    'default': 0,
    'eeg': 0,

    # a rarely used EEG channel
    'EEG Pz-Oz': 1, 

    # EOG channel, with good classification performance
    'EOG horizontal': 2, 
    'eog': 2, 
    
    'Resp oro-nasal': 3, 
    'resp': 3,

    'EMG submental': 4,
    'emg': 4,
    }

FD_B = {
    'training-set-fixed-scaling': 1,
    'default': 0,
}

SleepEOG = {
    'training-set-fixed-scaling': 1,
    'default': 0,
}

ExpEMG = {
    'training-set-fixed-scaling': 100,
    'default': 0,
}

TFC_Epilepsy = {
    'training-set-fixed-scaling': 1,
    'default': 0,
}


class bioFAME_data(Dataset):
    # Initialize your data.
    def __init__(
        self, 
        path, 
        filename='train.pt',
        channels=(1, 7),
        trim_end=3000,
        transforms=None,
        dataset_name=None,
        ):
        super(bioFAME_data, self).__init__()

        self.channels = channels
        if dataset_name is not None:
            self.dataset_name = globals()[dataset_name]
        else:
            self.dataset_name = globals()['SleepEDF_multi']

        if 'trim' in self.dataset_name.keys():
            self.trim_end = self.dataset_name['trim']
        else:
            self.trim_end = trim_end

        self._datasetpt = self._load_from_path(path, filename)
        pt_data = self._datasetpt['samples']
        pt_label = self._datasetpt['labels']

        # shuffle the dataset
        datapack = list(zip(pt_data, pt_label))
        np.random.shuffle(datapack)
        self.data, self.label = zip(*datapack)
        self.data, self.label = torch.stack(list(self.data), dim=0), torch.stack(list(self.label), dim=0)

        logger.info('Confirming data shape and label shape of {}'.format(dataset_name))
        logger.info('{} {}'.format(self.data.shape, self.label.shape))

        self.data, self.label = torch.Tensor(self.data), torch.Tensor(self.label)
        self.data = self.data.float()
        self.transforms = transforms

    def __getitem__(self, index):

        assert type(self.channels) == type(["A", "B"])
        '''specify input channel with list of strings'''
        channel_index = [self.dataset_name[chan_name] for chan_name in self.channels]
        data_scaling = self.dataset_name['training-set-fixed-scaling']

        data, label = data_scaling * self.data[index, channel_index, :self.trim_end], self.label[index]

        if len(data.shape) == 3:
            data = data[0, :, :]

        if self.transforms is not None:
            data = self.transforms(data)
            data = data.float()

        return data, label

    def __len__(self):
        return self.data.shape[0]

    def _load_from_path(self, path, filename='train.pt'):        
        return torch.load(os.path.join(path, filename))

    @staticmethod
    def _save_to_pt(data, label, save_path='', file_name=''):
        """How to generate a readable file by this class"""
        # assert X_tensor should be [batch x channel x length]

        save_dict = {}
        save_dict['samples'] = data
        save_dict['labels'] = label

        torch.save(save_dict, os.path.join(save_path, "{}.pt".format(file_name)))

