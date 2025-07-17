import torch
import numpy as np
import scipy.io as sio

from torch.utils.data import TensorDataset, DataLoader


def load_cost_test_data_raw(data_dir, env='in', return_tensor=True):

    data = sio.loadmat(data_dir + 'DATA_HtestF' + env + '_all.mat')['HF_all']

    if return_tensor is True:
        data_real = torch.tensor(np.real(data), dtype=torch.float32).view(data.shape[0], 32, 125)
        data_imag = torch.tensor(np.imag(data), dtype=torch.float32).view(data.shape[0], 32, 125)

        data = torch.cat((torch.unsqueeze(data_real, 1), torch.unsqueeze(data_imag, 1)), dim=1)

    return data



def load_cost_data_sparse(data_dir, env='in', return_tensor=True):

    x_train = sio.loadmat(data_dir + 'DATA_Htrain' + env + '.mat')['HT']
    x_val = sio.loadmat(data_dir + 'DATA_Hval' + env + '.mat')['HT']
    x_test = sio.loadmat(data_dir + 'DATA_Htest' + env + '.mat')['HT']

    if return_tensor is True:
        x_train = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], 2, 32, 32)
        x_val = torch.tensor(x_val, dtype=torch.float32).view(x_val.shape[0], 2, 32, 32)
        x_test = torch.tensor(x_test, dtype=torch.float32).view(x_test.shape[0], 2, 32, 32)

    return x_train, x_val, x_test


def get_cost_dataset(scenario='out', batch_size=200):
    general_path = './COST2100_dataset/'

    x_train, x_val, x_test = load_cost_data_sparse(data_dir=general_path, env=scenario)

    train_loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val), batch_size=batch_size, shuffle=False)

    x_test_raw = load_cost_test_data_raw(data_dir=general_path, env=scenario)

    test_loader = DataLoader(TensorDataset(x_test, x_test_raw), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
