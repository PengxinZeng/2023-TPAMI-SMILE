import os
import warnings

from PIL import Image
import numpy as np
import scipy.io as sio
import torchvision
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

from _Utils.Visualize import describ_distribution, describ_distribution_image, tsne
from utils import TT_split, normalize
import torch
import random
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
from _MainLauncher import path_operator


def trans(data, args):
    new_xs = []
    for x in data:
        x = torch.from_numpy(x)
        x = x.view((-1, 1, 1, x.shape[-1]))
        if args.normalize_type == 'dim_wise':
            mean, std = torch.mean(x, dim=0), torch.std(x, dim=0)
            std[std < torch.max(std) * 1e-6] = 1
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif args.normalize_type == 'sample_wise':
            mean, std = torch.mean(x), torch.std(x)
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif args.normalize_type == 'rescale_dim_wise':
            ma = torch.amax(x, dim=0)
            mi = torch.amin(x, dim=0)
            mean, std = (ma + mi) / 2, (ma - mi) / 2
            std[std < torch.max(std) * 1e-6] = 1
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif args.normalize_type == 'rescale_sample_wise':
            ma = torch.amax(x)
            mi = torch.amin(x)
            mean, std = (ma + mi) / 2, (ma - mi) / 2
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif args.normalize_type == 'None':
            pass
        else:
            raise NotImplementedError("")
        if args.withTanh:
            x = torch.nn.Tanh()(x)
        x = x.view((-1, x.shape[-1])).numpy()
        if args.DimensionalityReduction:
            if x.shape[1] != args.DimensionalityReduction:
                x = PCA(n_components=args.DimensionalityReduction).fit_transform(x)
        # if args.DimensionalityReduction:
        #     fv = np.load('/xlearning/pengxin/Codes/0831/RunSet0912_DR_warm/ --DimensionalityReduction 0 --EvalMulti 1 --VisualFreq 1 --WarmAll 999 --batch_size 64 --dataset Scene15 --normalize_type dim_wise --seed 1215 --train_epoch 300 --withTanh 1/NpPoints/Np299.npz', allow_pickle=False)['feature_vec']
        #     data[0] = fv[:len(data[0])]
        #     data[1] = fv[-len(data[1]):]
        new_xs.append(x)
    return new_xs


def load_data(args, neg_prop, aligned_prop, complete_prop, is_noise):
    """

    :param args:
    :param neg_prop:
    :param aligned_prop:
    :param complete_prop:
    :param is_noise:
    :return:
        train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, divide_seed, mask, train_sample_num
        mask: shape (n, 2), {0:缺失, 1}
    """
    dataset = args.dataset
    all_data = []
    train_pairs = []
    label = []
    # if dataset == 'Scene15':
    #     mat = sio.loadmat('/xlearning/pengxin/Data/ImageClassification/Scene15.mat')
    # else:
    #     mat = sio.loadmat('/xlearning/mouxing/datasets/MvData/' + dataset + '.mat')
    mid_name = 'Data'

    mat = sio.loadmat(os.path.join(path_operator.home, mid_name, dataset + '.mat'))

    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y']) - 1
    elif dataset == 'Caltech101':
        data = mat['X'][0][3:5]
        label = np.squeeze(mat['Y'])
    elif dataset == 'Caltech101-20' or dataset == 'Caltech101-all':
        data = mat['X'][0][3:5]
        label = np.squeeze(mat['Y']) - 1
    elif dataset == 'Anchor_Caltech101':
        data = []
        data.append(mat['Anchor'][0][3].T)
        data.append(mat['Anchor'][0][4].T)
        label = np.squeeze(mat['Y'])

    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples

        # data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        # data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        data.append(np.vstack((mat['x_train'][0], mat['x_test'][0])))
        data.append(np.vstack((mat['x_train'][1], mat['x_test'][1])))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))

    elif dataset == 'NoisyMNIST30000':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y']) - 1
    elif dataset == 'NoisyMNIST':
        data = []
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['trainLabel']) - 1

    # deep features of Caltech101
    # elif dataset == 'caltech_vgg16' or dataset == 'caltech_vgg19':
    #     data = []
    #     label = np.squeeze(mat['gt'])
    #     data.append(mat['X'][0][0].T)
    #     data.append(mat['X'][0][1].T)
    elif dataset == '2view-caltech101-8677sample':
        data = []
        label = np.squeeze(mat['gt']) - 1
        data.append(mat['X'][0][1].T)
        data.append(mat['X'][0][0].T)

    elif dataset == 'MNIST-USPS':
        data = [mat['X1'], mat['X2']]
        label = np.squeeze(mat['Y'])

    # deep features of Animal
    elif dataset == 'AWA-7view-10158sample':
        data = []
        label = np.squeeze(mat['gt']) - 1
        data.append(mat['X'][0][5].T)
        data.append(mat['X'][0][6].T)
    elif dataset == 'cub_googlenet_doc2vec_c10':
        data = []
        label = np.squeeze(mat['gt']) - 1
        data.append(mat['X'][0][0])
        data.append(mat['X'][0][1])
    else:
        raise NotImplementedError('')

    data = trans(data, args)

    # for ind, di in enumerate(data):
    #     describ_distribution_image([di[y == label] for y in np.unique(label)],
    #                                fig_path='../_fig/data{}_dis_img.jpg'.format(ind))
    describ_distribution(data, fig_path='../_fig/distribution.jpg')

    divide_seed = random.randint(1, 1000)
    train_idx, test_idx = TT_split(len(label), 1 - aligned_prop, divide_seed)
    train_label, test_label = label[train_idx], label[test_idx]
    train_X, train_Y, test_X, test_Y = data[0][train_idx], data[1][train_idx], data[0][test_idx], data[1][test_idx]
    train_sample_num = len(train_X)
    # Use test_prop*sizeof(all data) to train the MvCLN, and shuffle the rest data to simulate the unaligned data.
    # Note that, MvCLN establishes the correspondence of the all data rather than the unaligned portion in the testing.
    # When test_prop = 0, MvCLN is directly performed on the all data without shuffling.
    if aligned_prop == 1:
        all_data.append(train_X)
        all_data.append(train_Y)
        all_label, all_label_X, all_label_Y = train_label, train_label, train_label
    else:
        # shuffle_idx = random.sample(range(len(test_Y)), len(test_Y))
        # test_Y = test_Y[shuffle_idx]
        # test_label_X, test_label_Y = test_label, test_label[shuffle_idx]
        warnings.warn('You did not shuffle the unaligned data')
        test_label_X, test_label_Y = test_label, test_label
        all_data.append(np.concatenate((train_X, test_X)))
        all_data.append(np.concatenate((train_Y, test_Y)))
        all_label = np.concatenate((train_label, test_label))
        all_label_X = np.concatenate((train_label, test_label_X))
        all_label_Y = np.concatenate((train_label, test_label_Y))

    test_mask = get_sn(2, len(test_label), 1 - complete_prop)
    if aligned_prop == 1.:
        mask = test_mask
    else:
        identy_mask = np.ones((len(train_label), 2))
        mask = np.concatenate((identy_mask, test_mask))

    # pair construction. view 0 and 1 refer to pairs constructed for training. noisy and real labels refer to 0/1 label of those pairs
    if aligned_prop == 1.:
        valid_idx = np.logical_and(mask[:, 0], mask[:, 1])
    else:
        valid_idx = np.ones_like(train_label).astype(np.bool_)
    view0, view1, noisy_labels, real_labels, _, _ = \
        get_pairs(train_X[valid_idx], train_Y[valid_idx], neg_prop, train_label[valid_idx])

    count = 0
    for i in range(len(noisy_labels)):
        if noisy_labels[i] != real_labels[i]:
            count += 1
    print('noise rate of the constructed neg. pairs is ', round(count / (len(noisy_labels) - len(train_X) + 1e-2), 2))

    if is_noise:  # training with noisy negative correspondence
        print("----------------------Training with noisy_labels----------------------")
        train_pair_labels = noisy_labels
    else:  # training with gt negative correspondence
        print("----------------------Training with real_labels----------------------")
        train_pair_labels = real_labels
    train_pairs.append(view0.T)
    train_pairs.append(view1.T)
    train_pair_real_labels = real_labels

    return train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, divide_seed, mask, train_sample_num


def get_pairs(train_X, train_Y, neg_prop, train_label):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []
    # construct pos. pairs
    for i in range(len(train_X)):
        view0.append(train_X[i])
        view1.append(train_Y[i])
        labels.append(1)
        real_labels.append(1)
        class_labels0.append(train_label[i])
        class_labels1.append(train_label[i])
    # construct neg. pairs by taking each sample in view0 as an anchor and randomly sample neg_prop samples from view1,
    # which may lead to the so called noisy labels, namely, some of the constructed neg. pairs may in the same category.
    for j in range(len(train_X)):
        neg_idx = random.sample(range(len(train_Y)), neg_prop)
        for k in range(neg_prop):
            view0.append(train_X[j])
            view1.append(train_Y[neg_idx[k]])
            labels.append(0)
            class_labels0.append(train_label[j])
            class_labels1.append(train_label[neg_idx[k]])
            if train_label[j] != train_label[neg_idx[k]]:
                real_labels.append(0)
            else:
                real_labels.append(1)

    labels = np.array(labels, dtype=np.int64)
    real_labels = np.array(real_labels, dtype=np.int64)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int64), np.array(class_labels1, dtype=np.int64)
    # view0 = torch.concat(view0)
    # view1 = torch.concat(view1)
    view0 = np.array(view0, dtype=np.float32)
    view1 = np.array(view1, dtype=np.float32)
    return view0, view1, labels, real_labels, class_labels0, class_labels1


def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 4.3 of the paper
    :return:Sn
    """
    # if missing_rate == 1:
    #     mt = np.ones((alldata_len, 2), dtype=int)
    #     ind = np.arange(alldata_len)
    #     np.random.shuffle(ind)
    #     ind0 = ind[:int(alldata_len/2)]
    #     ind1 = ind[int(alldata_len/2):]
    #     mt[ind0, 0] = 0
    #     mt[ind1, 1] = 0
    #     return mt
    missing_rate = missing_rate / 2
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        if len(view_preserve[0]) == 1:
            vp = np.zeros((len(view_preserve), 2))
            vp[:, 0] = view_preserve[:, 0]
            view_preserve = vp
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix


class getDataset(Dataset):
    def __init__(self, data, labels, real_labels):
        self.data = data
        self.labels = labels
        self.real_labels = real_labels

    def __getitem__(self, index):
        fea0, fea1 = (torch.from_numpy(self.data[0][:, index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][:, index])).type(torch.FloatTensor)
        fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        label = np.int64(self.labels[index])
        if len(self.real_labels) == 0:
            return fea0, fea1, label
        real_label = np.int64(self.real_labels[index])
        return fea0, fea1, label, real_label

    def __len__(self):
        return len(self.labels)


class getAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1, mask, train_sample_num, transforms=None):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1
        self.mask = mask == 1
        self.train_sample_num = train_sample_num
        if transforms is None:
            transforms = [None, None]
        self.transforms = transforms

    def __getitem__(self, index):
        # print(self.data[0].shape)
        # ax = [len(self.data[0].shape) -1] + list( np.arange(len(self.data[0].shape) -1))
        # fea0 = np.transpose(self.data[0], axes=ax)
        # fea1 = np.transpose(self.data[1], axes=ax)
        fea0, fea1 = (torch.from_numpy(self.data[0][index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][index])).type(torch.FloatTensor)
        # fea0, fea1 = fea0.unsqueeze(0), fea1.unsqueeze(0)
        # if self.transforms[0] is not None:
        #     # noinspection PyCallingNonCallable
        #     fea0 = self.transforms[0](fea0)
        # if self.transforms[1] is not None:
        #     # noinspection PyCallingNonCallable
        #     fea1 = self.transforms[1](fea1)
        label = np.int64(self.labels[index])
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        mask = self.mask[index]
        is_pair = self.train_sample_num > index
        return fea0, fea1, class_labels0, class_labels1, mask, is_pair, index

    def __len__(self):
        return len(self.labels)


def loader(train_bs, neg_prop, aligned_prop, complete_prop, is_noise, dataset):
    """
    :param train_bs: batch size for training, default is 1024
    :param neg_prop: negative / positive pairs' ratio
    :param aligned_prop: known aligned proportions for training SURE
    :param complete_prop: known complete proportions for training SURE
    :param is_noise: training with noisy labels or not, 0 --- not, 1 --- yes
    :param dataset: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing MvCLN
    """
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, \
    divide_seed, mask = load_data(dataset, neg_prop, aligned_prop, complete_prop, is_noise)
    train_pair_dataset = getDataset(train_pairs, train_pair_labels, train_pair_real_labels)
    all_dataset = getAllDataset(all_data, all_label, all_label_X, all_label_Y, mask)

    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    all_loader = DataLoader(
        all_dataset,
        batch_size=1024,
        shuffle=True
    )
    return train_pair_loader, all_loader, divide_seed


def get_dataset_fair_clustering_mouxin(neg_prop, aligned_prop, complete_prop, is_noise, args, **kwargs):
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, \
    divide_seed, mask, train_sample_num = load_data(args, neg_prop, aligned_prop, complete_prop, is_noise)

    all_dataset = getAllDataset(all_data, all_label, all_label_X, all_label_Y, mask, train_sample_num, **kwargs)
    class_num = len(np.unique(all_label_X))
    return all_dataset, class_num
