import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import random
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
from Clustering import Clustering


def TT_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    :return:
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1 - test_prop) * n_all).astype(np.int)
    train_idx = random_idx[0:train_num]
    test_num = np.floor(test_prop * n_all).astype(np.int)
    test_idx = random_idx[-test_num:]
    return train_idx, test_idx


def svm_classify(data, data_gt, label, test_prop, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    seed = random.randint(0, 1000)
    train_idx, test_idx = TT_split(data.shape[1], test_prop, seed)
    train_data = np.concatenate([data[0][train_idx], data[1][train_idx]], axis=1)
    test_data = np.concatenate([data_gt[0][test_idx], data_gt[1][test_idx]], axis=1)
    test_label = label[test_idx]
    train_label = label[train_idx]

    # print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)

    return test_acc


def knn(data, label, test_prop, k):
    seed = random.randint(0, 1000)
    train_idx, test_idx = TT_split(data.shape[0], test_prop, seed)
    train_data = data[train_idx, :]
    test_data = data[test_idx, :]
    test_label = label[test_idx]
    train_label = label[train_idx]

    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(train_data, train_label)
    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)

    return test_acc

if __name__ == '__main__':

    mat = sio.loadmat('./exp/representation/1vs1/128d/batchsize_1024/NoisyMNIST-30000_robust_Divide=972_0.9636.mat')
    data = []
    data.append(np.array(mat['X1']))
    data.append(np.array(mat['X2']))
    label = np.squeeze(mat['label'])
    # y_pred, ret = Clustering(data, label)

    test_prop = 0.2
    for test_prop in [0.2, 0.5, 0.8]:
        acc = 0
        for i in range(20):
            tmp_acc = np.round(svm_classify(np.array(data), label, test_prop=test_prop, C=0.01), 4)
            acc += tmp_acc
            # print('Time {}, test_prop = {}, acc = {}'.format(i, test_prop, tmp_acc))
        print('Test_prop = {}, Avg acc = {}'.format(test_prop, np.round(acc / 20, 4)))