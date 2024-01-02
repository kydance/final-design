import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
from hdf5storage import loadmat as loadmat_73

class Data:
    class __Data(Dataset):
        def __init__(self, dataset_dir, dataset_name='xiongan', train=True, dataset_suffix=".mat"):
            # super(__Data, self).__init__()
            mat_loader = loadmat_73(os.path.join(dataset_dir, dataset_name + dataset_suffix))

            self.map = np.array(mat_loader['map'], dtype=np.float32)
            self.d = np.array(mat_loader['d'], dtype=np.float32)[0:50]
            self.size = self.map.shape
            if train:
                self.data = np.asarray(mat_loader['train_data'], dtype=np.float32)[:, 0:50]
            else:
                self.data = np.asarray(mat_loader['ori_data'], dtype=np.float32)[: ,: , 0:50]
                self.data = np.reshape(self.data, [self.size[0] * self.size[1], -1])

            self.bands = self.data.shape[-1]
            self.data = self.__normalize(self.data)
            self.d = self.__normalize(self.d)

        def __len__(self):
            return int(self.data.shape[0])

        def __getitem__(self, item):
            return self.data[item, :]

        def __normalize(self, data):
            """TODO 耗时太多，需优化"""
            return (data - data.min()) / (data.max() - data.min()) * 2

    def __init__(self, batch_size=128, shuffle=False, data_path=None, data_name='aerorit'):
        self.dataset = self.__Data(data_path, data_name)
        self.train_loader = DataLoader(self.dataset, batch_size, shuffle)
        self.test_loader = DataLoader(self.__Data(data_path, data_name, False), batch_size, False)
