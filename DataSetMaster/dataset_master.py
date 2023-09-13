import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from DataSetMaster.data_loader import get_dataset_fair_clustering_mouxin, get_sn
from DataSetMaster.dataset import get_loader_new
from _MainLauncher import path_operator
from _Utils import DirectoryOperator
from _Utils.Visualize import describ_distribution


class GetDataset(Dataset):
    def __init__(self, ds, args):
        self.datasets = ds
        super(GetDataset, self).__init__()
        self.is_pair = torch.zeros(len(ds[0].data), dtype=torch.int)

        # align
        aligned_len = int(len(self.is_pair) * args.aligned_prop)
        aligned = np.arange(len(self.is_pair))
        np.random.shuffle(aligned)
        self.is_pair[aligned[:aligned_len]] = 1
        self.is_pair = self.is_pair == 1
        test_mask = get_sn(2, len(self.is_pair) - aligned_len, 1 - args.complete_prop)
        if aligned_len == 0:
            mask = test_mask
        else:
            mask = np.ones((len(self.is_pair), 2), dtype=int)
            mask[aligned[aligned_len:]] = test_mask
            # mask = np.concatenate((np.ones((aligned_len, 2)), test_mask), axis=0)
        self.mask = np.asarray(mask, dtype=int)
        self.mask = self.mask == 1
        # print('mask.sum() == {}'.format(mask.sum()))
        # print('self.is_pair.sum() == {}'.format(self.is_pair.sum()))
        # print('(self.datasets[0].targets-self.datasets[0].targets).sum() == {}'.format(
        # (self.datasets[0].targets-self.datasets[0].targets).sum()))
        # print()

    def __getitem__(self, index):
        xs = []
        ys = []
        for dset in self.datasets:
            x = dset.data[index % len(dset)]
            y = int(dset.targets[index % len(dset)])
            xs.append(x)
            ys.append(y)
        mask = self.mask[index]
        is_pair = self.is_pair[index]
        return *xs, *ys, mask, is_pair, index

    def __len__(self):
        return np.max([len(dset) for dset in self.datasets])


def get_dataset_px(args):
    print(path_operator.get_dataset_path('MNIST', level=2),)
    if args.dataset == 'MNISTUSPS':
        d0 = torchvision.datasets.MNIST(
            train=args.MnistTrain,
            download=False,
            root=path_operator.get_dataset_path('MNIST', level=2),
            transform=None)
        d1 = torchvision.datasets.USPS(
            train=True,
            download=False,
            root=path_operator.get_dataset_path('USPS', level=2),
            transform=None)
        if args.DataSource:
            data = np.load(args.DataSource)
            d0.data = torch.from_numpy(data['d0_data'])
            d0.targets = data['d0_targets']
            d1.data = torch.from_numpy(data['d1_data'])
            d1.targets = data['d1_targets']
        else:
            def trans(x):
                # w = int(np.sqrt(x.shape[1]))
                # if not (w == 16 or w == 28):
                #     raise NotImplementedError('not (w == 16 or w == 28)')
                # x = np.reshape(x, (-1, w, w))
                res = []
                for x0 in x:
                    x0 = Image.fromarray(x0)
                    x0 = torchvision.transforms.Resize(28)(x0)
                    x0 = torchvision.transforms.ToTensor()(x0)
                    # x0 = torchvision.transforms.Normalize([0.5], [0.5])(x0)
                    res.append(x0)
                x = torch.concat(res)
                if args.Image01NoTanh:
                    mi, ma = torch.amin(x), torch.amax(x)
                    x = torchvision.transforms.Normalize([(mi + ma) / 2], [(ma - mi) / 2])(x)
                else:
                    mean, std = torch.mean(x), torch.std(x)
                    # x = torchvision.transforms.Normalize([255/2], [255/2])(x)
                    x = torchvision.transforms.Normalize([mean], [std])(x)
                    x = torch.nn.Tanh()(x)
                x = torch.unsqueeze(x, dim=1)
                return x

            x1 = trans(d0.data.numpy())
            x2 = trans(d1.data)
            if args.Image01AndTanh:
                x_ = torch.cat([x1, x2])
                mean, std = torch.mean(x_), torch.std(x_)
                x_ = torchvision.transforms.Normalize([mean], [std])(x_)
                x_ = torch.nn.Tanh()(x_)
                x1 = x_[:len(x1)]
                x2 = x_[-len(x2):]
            d0.data = x1
            d1.data = x2

            def re_sample(dx):
                classes = np.unique(dx.targets)
                num_per_class = int(args.MaxSampleNum / len(classes))
                ind = []
                for y in classes:
                    nd = np.arange(len(dx.targets))[dx.targets == y]
                    np.random.shuffle(nd)
                    ind.extend(list(nd[:num_per_class]))
                dx.data = dx.data[ind]
                dx.targets = np.asarray(dx.targets)[ind]

            if len(d0.data) > args.MaxSampleNum:
                re_sample(d0)
            if len(d1.data) > args.MaxSampleNum:
                re_sample(d1)

            # def re_sample(dx):
            #     classes = np.unique(dx.targets)
            #     num_per_class = int(args.MaxSampleNum / len(classes))
            #     ind = []
            #     for y in classes:
            #         nd = np.arange(len(dx.targets))[dx.targets == y]
            #         np.random.shuffle(nd)
            #         ind.extend(list(nd[:num_per_class]))
            #     dx.data = dx.data[ind]
            #     dx.targets = np.asarray(dx.targets)[ind]
            #
            # if len(d0.data) > args.MaxSampleNum:
            #     re_sample(d0)
            # if len(d1.data) > args.MaxSampleNum:
            #     re_sample(d1)
            # d0.data, d1.data = trans([d0.data, d1.data], args)
        d0.data = d0.data.view((len(d0.data), -1))
        d1.data = d1.data.view((len(d1.data), -1))
        np_dir = '../SampleCache/Np'
        DirectoryOperator.FoldOperator(directory=np_dir).make_fold()
        np.savez(np_dir, d0_data=d0.data, d0_targets=d0.targets, d1_data=d1.data, d1_targets=d1.targets)
        class_num = 10
        ds = GetDataset([d0, d1], args)
    else:
        raise NotImplementedError('')
    describ_distribution([d.data.numpy() for d in ds.datasets], fig_path='../_fig/distribution.jpg')
    return ds, class_num


def get_dataloader_mouxin(args, **kwargs):
    # print('getting data {}'.format(args.dataset))
    if args.dataset == 'MNISTUSPS':
        data_set, class_num = get_dataset_px(args=args)
    else:
        data_set, class_num = get_dataset_fair_clustering_mouxin(args=args, **kwargs)
    train_loader, test_loader = get_loader_new(data_set, args.batch_size)
    # print('got data {}'.format(args.dataset))
    return train_loader, test_loader, class_num
