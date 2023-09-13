import numpy as np
import scipy.io as scio


def load_mat(src):
    return scio.loadmat(src)


def save_mat(dst, dic):
    print('SavingMat')
    scio.savemat(dst, dic)


def test_save_load():
    data = {'X': np.random.random((2, 500, 784))}
    dst = r'D:\Pengxin\Temp\test.mat'
    save_mat(dst=dst, dic=data)
    data_loaded = load_mat(src=dst)
    print(np.sum(data['X'] - data_loaded['X']))
    return
