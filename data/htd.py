import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
from hdf5storage import loadmat as loadmat_73

def normalize(data):
    """TODO 耗时太多，需优化"""
    data = (data - data.min()) / (data.max() - data.min()) * 2
    return data

class InternalData(Dataset):
    def __init__(self, dataset_dir, dataset_name='xiongan', train=True, dataset_suffix="_post.mat"):
        super(InternalData, self).__init__()
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
        self.data = normalize(self.data)
        self.d = normalize(self.d)

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, item):
        return self.data[item, :]

def get_dataloader(batch_size=128, shuffle=False, data_path=None, data_name='aerorit'):
        dataset = InternalData(data_path, data_name)
        train_loader = DataLoader(dataset, batch_size, shuffle)
        test_loader = DataLoader(InternalData(data_path, data_name, False), batch_size, False)
        return (dataset, train_loader, test_loader)

class Data:
    def __init__(self, batch_size=128, shuffle=False, data_path=None, data_name='aerorit'):
        self.dataset = InternalData(data_path, data_name)
        self.train_loader = DataLoader(self.dataset, batch_size, shuffle)
        self.test_loader = DataLoader(InternalData(data_path, data_name, False), batch_size, False)


# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from hdf5storage import loadmat as loadmat_73


# def normalize(data):
#     data = (data - data.min()) / (data.max() - data.min()) * 2

#     return data

# class Data(Dataset):
#     def __init__(self, name='xiongan', train=True):
#         super(Data, self).__init__()
#         if name == 'xiongan':
#             mat_loader = loadmat_73('/home/lutianen/data/htd/xiongan_post.mat')
#         elif name == 'aerorit':
#             mat_loader = loadmat_73('/home/lutianen/data/htd/aerorit_post.mat')
#         else:
#             mat_loader = loadmat_73('/home/lutianen/data/htd/aerorit_post.mat')
#         self.map = np.array(mat_loader['map'], dtype=np.float32)
#         self.d = np.array(mat_loader['d'], dtype=np.float32)[0:50]
#         self.size = self.map.shape
#         if train:
#             self.data = np.asarray(mat_loader['train_data'], dtype=np.float32)[:, 0:50]
#         else:
#             self.data = np.asarray(mat_loader['ori_data'], dtype=np.float32)[: ,: , 0:50]
#             self.data = np.reshape(self.data, [self.size[0] * self.size[1], -1])
#         self.bands = self.data.shape[-1]
#         self.data = normalize(self.data)
#         self.d = normalize(self.d)

#     def __len__(self):
#         return int(self.data.shape[0])

#     def __getitem__(self, item):
#         return self.data[item, :]


# def get_dataloader(batch_size=128, shuffle=False, data_name='aerorit'):
#     dataset = Data(data_name)
#     dataloader = DataLoader(dataset, batch_size, shuffle)
#     test_loader = DataLoader(Data(data_name, False), batch_size, False)

#     return (dataset, dataloader, test_loader)

