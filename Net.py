import math
import os
import time
import warnings

import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from DistComput import get_dist_release
from _Utils.Calculator import get_nearest_k
from _Utils.Logs import update_log
from _Utils.Scatter import visualize2
from _Utils.Visualize import visualize, visual_matrix_console, visualize_image, plot_heat_map
from _Utils import TimeOperator, DirectoryOperator
from DataSetMaster.dataset import get_clusters
from classification import svm_classify
from evaluate import UMAP, evaluate2
import evaluate
import faiss
from sklearn import metrics
from munkres import Munkres
from figures.ScatterMaster import visual_image_scatter
import scipy.io as sio


def show_distribution_ct(type_vec, group_vec, pred_vec, class_num, group_num):
    v = np.zeros((class_num, class_num, group_num), dtype=int)
    for t, c, g in zip(type_vec, pred_vec, group_vec):
        v[t, c, g] += 1
    visual_matrix_console(x=v)


def kmeans(feature_vec, class_num):
    d = feature_vec.shape[1]
    kmeans = faiss.Clustering(d, class_num)
    kmeans.verbose = False
    kmeans.niter = 300
    kmeans.nredo = 10
    # kmeans.spherical = True
    # if LimitKmeans:
    #     kmeans.max_points_per_centroid = 1000
    #     kmeans.min_points_per_centroid = 10
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = True
    cfg.device = 0
    index = faiss.GpuIndexFlatL2(res, d, cfg)
    # print(feature_vec.shape)
    kmeans.train(feature_vec, index)
    centroids = faiss.vector_to_array(kmeans.centroids).reshape(class_num, d)
    return centroids


def show_distribution(cluster_vec, group_vec, class_num, group_num):
    for it in np.arange(group_num):
        print('{:4d}, '.format(it), end='')
    print('')
    cluster_group = torch.zeros((class_num, group_num), dtype=torch.int)
    for i, j in zip(cluster_vec, group_vec):
        cluster_group[i, j] += 1
    # cluster_group = cluster_group[torch.argsort(torch.sum(cluster_group, dim=1))]
    for line in cluster_group:
        print('{:4d}: '.format(torch.sum(line)), end='')
        for it in line:
            print('{:4d}, '.format(it), end='')
        print('')


def save_checkpoint(state, epoch):
    """
    it has been trained for *epoch* epochs
    """
    filename = 'Epoch{:03d}.checkpoint'.format(epoch)
    checkpoint_dir = os.path.join(
        os.path.dirname(os.getcwd()),
        'Checkpoints',
        filename
    )
    DirectoryOperator.FoldOperator(directory=checkpoint_dir).make_fold()
    if os.path.exists(checkpoint_dir):
        warnings.warn('Checkpoint exist and been replaced.({})'.format(checkpoint_dir))
    print('Save check point into {}'.format(checkpoint_dir))
    torch.save(state, checkpoint_dir)


def get_ffn(dims, last_layers=None, with_bn=False, drop_out=0):
    layers = []
    for ind in range(len(dims) - 1):
        in_dim = dims[ind]
        out_dim = dims[ind + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        if with_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        if drop_out:
            layers.append(nn.Dropout(drop_out))
    if last_layers is not None:
        layers.extend(last_layers)
    return nn.Sequential(*layers)


def get_cov(dims, strides, last_layers=None, with_bn=False, drop_out=0):
    layers = []
    for ind in range(len(dims) - 1):
        in_dim = dims[ind]
        out_dim = dims[ind + 1]
        stride = strides[ind]
        # layers.append(nn.Linear(in_dim, out_dim))
        if stride >= 0:
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1))
        else:
            layers.append(nn.ConvTranspose2d(
                in_dim, out_dim, kernel_size=3, stride=-stride, padding=1, output_padding=0 if stride == -1 else 1))
        if with_bn:
            # layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU())
        if drop_out:
            layers.append(nn.Dropout(drop_out))
    if last_layers is not None:
        layers.extend(last_layers)
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self, args, in_dims, class_num, group_num):
        super(Net, self).__init__()
        self.encoder_adaption = nn.ModuleList([
            get_ffn([in_dims[i], 1024], with_bn=args.BatchNormType[0] == '1', drop_out=args.Dropout)
            for i in range(group_num if args.GroupWiseLayer[0] == '1' else 1)])
        self.encoder = nn.ModuleList([
            get_ffn([1024, 1024, 512], with_bn=args.BatchNormType[1] == '1', drop_out=args.Dropout)
            for _ in range(group_num if args.GroupWiseLayer[1] == '1' else 1)])
        if args.representation_dim == 0:
            args.representation_dim = class_num
        self.class_num = class_num
        self.group_num = group_num
        self.pred_cac = None
        self.pred_center_cac = None

        if args.ElActivationType == 'None':
            el_activation_ = []
        elif args.ElActivationType == 'Normalize':
            el_activation_ = []
        elif args.ElActivationType == 'BnNormalize':
            el_activation_ = [nn.BatchNorm1d(args.representation_dim)]
        elif args.ElActivationType == 'BnReNormalize':
            el_activation_ = [nn.BatchNorm1d(args.representation_dim), nn.ReLU()]
        elif args.ElActivationType == 'BnRe':
            el_activation_ = [nn.BatchNorm1d(args.representation_dim), nn.ReLU()]
        else:
            raise NotImplementedError('')
        self.el_activation_ = el_activation_
        self.encoder_linear = nn.ModuleList([
            get_ffn([512, 256], with_bn=args.BatchNormType[2] == '1', drop_out=args.Dropout,
                    last_layers=[nn.Linear(256, args.representation_dim)] + self.el_activation_)
            for _ in range(group_num if args.GroupWiseLayer[2] == '1' else 1)])
        dec_in = args.representation_dim
        if args.McDecoder:
            dec_in *= group_num
        self.dec_in = dec_in
        self.decoder_linear = nn.ModuleList([
            get_ffn([self.dec_in, 256, 512], with_bn=args.BatchNormType[3] == '1', drop_out=args.Dropout)
            for _ in range(group_num if args.GroupWiseLayer[3] == '1' else 1)])

        if args.ActivationType == 'None':
            final_activation_ = []
        elif args.ActivationType == 'Sigmoid':
            final_activation_ = [nn.Sigmoid()]
        elif args.ActivationType == 'Tanh':
            final_activation_ = [nn.Tanh()]
        else:
            raise NotImplementedError('')
        self.final_activation_ = final_activation_
        self.decoder = nn.ModuleList([
            get_ffn([512, 1024, 1024], with_bn=args.BatchNormType[4] == '1', drop_out=args.Dropout)
            for _ in range(group_num if args.GroupWiseLayer[4] == '1' else 1)])

        self.decoder_adaption = nn.ModuleList([
            get_ffn([], last_layers=[nn.Linear(1024, in_dims[i])] + self.final_activation_)
            for i in range(group_num if args.GroupWiseLayer[5] == '1' else 1)])
        self.args = args
        self.in_dims = in_dims

    # def update_cluster_center(self, center):
    #     self.cluster_centers = F.normalize(torch.from_numpy(center), dim=1).cuda()
    def forward(self, x, **kwargs):
        return self.decode(self.encode([x]))

    def encode(self, xs: list):
        hs = []
        for g, x in enumerate(xs):
            if self.args.noise_type == 'None':
                pass
            elif self.args.noise_type == 'Drop':
                x = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) < self.args.noise_weight).type_as(x)
            elif self.args.noise_type == 'Add':
                x = x + Variable(x.data.new(x.size()).normal_(0, self.args.noise_weight)).type_as(x)
            else:
                raise NotImplementedError('')
            if len(x) != 0:
                if len(x) == 1:
                    x = torch.concat([x, x])
                # print(x.shape)
                # x = x.view((len(x), -1))
                # print(x.shape)
                x = self.encoder_adaption[g if self.args.GroupWiseLayer[0] == '1' else 0](x)
                x = self.encoder[g if self.args.GroupWiseLayer[1] == '1' else 0](x)
                x = self.encoder_linear[g if self.args.GroupWiseLayer[2] == '1' else 0](x)
                if len(x) == 1:
                    x = x[[0]]
                if self.args.ElActivationType in ['Normalize', 'BnNormalize', 'BnReNormalize']:
                    x = F.normalize(x, dim=1)
            else:
                x = torch.zeros([0, self.args.representation_dim], device=torch.device('cuda:0'))
            hs.append(x)
        return hs

    def soft_ass(self, h, centroids):
        if self.args.ElActivationType in ['Normalize', 'BnNormalize', 'BnReNormalize']:
            return h @ centroids.T
        else:
            dst = torch.cdist(h, centroids)
            # return (torch.mean(dst) - dst) / (torch.amax(dst) - torch.amin(dst)) * 2
            return -dst / 2

    # def encode_class(self, hs):
    #     cs = []
    #     for h in hs:
    #         c = h @ self.cluster_centers.T
    #         cs.append(c)
    #     return cs

    def decode(self, hs):
        xs = []
        for g, h in enumerate(hs):
            if self.args.McDecoder:
                h = torch.cat(hs, dim=1)
            if len(h) != 0:
                if len(h) == 1:
                    h = torch.concat([h, h])
                h = self.decoder_linear[g if self.args.GroupWiseLayer[3] == '1' else 0](h)
                h = self.decoder[g if self.args.GroupWiseLayer[4] == '1' else 0](h)
                h = self.decoder_adaption[g if self.args.GroupWiseLayer[5] == '1' else 0](h)
                if len(h) == 1:
                    h = h[[0]]
            else:
                h = torch.zeros([0, self.in_dims[g]], device=torch.device('cuda:0'))
            xs.append(h)
        return xs

    def run(self, epochs, train_dataloader, test_dataloader, args):
        # if args.loss_self_cons:
        #     clusters = get_clusters(args=args)

        optimizer_g = torch.optim.Adam(
            self.parameters(),
            lr=args.LearnRate,
            betas=(args.betas_a, args.betas_v),
            weight_decay=args.WeightDecay
        )
        mse_loss = nn.MSELoss().cuda()
        timer_all = TimeOperator.Timer()
        timer_train = TimeOperator.Timer()
        timer_save = TimeOperator.Timer()
        ce_loss = nn.CrossEntropyLoss().cuda()

        type_detail_shown = False
        start_epoch = 0
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                # if args.gpu is None:
                #     checkpoint = torch.load(args.resume)
                # else:
                #     # Map model to be loaded to specified single gpu.
                #     loc = 'cuda:{}'.format(args.gpu)
                #     checkpoint = torch.load(args.resume, map_location=loc)
                start_epoch = checkpoint['epoch']
                self.load_state_dict(checkpoint['state_dict'])
                optimizer_g.load_state_dict(checkpoint['optimizer']['optimizer_g'])
                # self.__dict__ = checkpoint['self_dic']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
                # self.args = args
                # warnings.warn('This is not equal to start from the beginning due to different rands states.')
                #
            else:
                raise NotImplementedError("=> no checkpoint found at '{}'".format(args.resume))
        if args.CodeTest:
            args.train_epoch = start_epoch + 1
            epochs = start_epoch + 1
        best_acc = 0
        for epoch in range(start_epoch, epochs):
            if (epoch + 1) <= args.LearnRateWarm:
                lr = args.LearnRate * (epoch + 1) / args.LearnRateWarm
            else:
                if args.LearnRateDecayType == 'None':
                    lr = args.LearnRate
                elif args.LearnRateDecayType == 'Exp':
                    lr = args.LearnRate * ((1 + 10 * (epoch + 1 - args.LearnRateWarm) / (
                            args.train_epoch - args.LearnRateWarm)) ** -0.75)
                elif args.LearnRateDecayType == 'Cosine':
                    lr = args.LearnRate * 0.5 * (1. + math.cos(
                        math.pi * (epoch + 1 - args.LearnRateWarm) / (args.train_epoch - args.LearnRateWarm)))
                else:
                    raise NotImplementedError('args.LearnRateDecayType')
            if lr != args.LearnRate:
                def adjust_learning_rate(optimizer):
                    print('adjust_learning_rate: {}'.format(lr))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                adjust_learning_rate(optimizer_g)

            timer_all_time = time.time()
            # inf_t = time.time()
            # print('start epoch {}'.format(epoch))
            self.eval()
            feature_vec, type_vec, group_vec = [], [], []
            feature_vec_cluster = []
            group_vec_cluster = []
            feature_vec_classification = []
            type_vec_cluster = []
            data_vec = []
            is_pair_all = []
            timer_infer_data = TimeOperator.Timer()
            rnmse_vec = [[], []]  # mask = 0 1
            with torch.no_grad():
                inf_data_t = time.time()
                for (fea0, fea1, class_labels0, class_labels1, mask, is_pair, index) in test_dataloader:
                    timer_infer_data.update(time.time() - inf_data_t)
                    # timer_infer_data.show(prefix='InferDataTime', total_count=len(test_dataloader),
                    #                       print_end_time=False)
                    fea0 = fea0.cuda()
                    fea1 = fea1.cuda()
                    if args.Rev:
                        h1, h0 = self.encode([fea0, fea1])
                        if args.SingleView != -1:
                            for v in range(len(mask[0])):
                                if v != 1 - args.SingleView:
                                    mask[:, v] = 0
                    else:
                        h0, h1 = self.encode([fea0, fea1])
                        if args.SingleView != -1:
                            for v in range(len(mask[0])):
                                if v != args.SingleView:
                                    mask[:, v] = 0
                    cluster_h0 = h0[mask[:, 0] == 1]
                    cluster_h1 = h1[mask[:, 1] == 1]
                    # if args.SingleView != -1:
                    #     mask[:, args.SingleView] = 0
                    #     # if args.SingleView == 0:
                    #     #     cluster_h1 = cluster_h1[[]]
                    #     #     class_labels1 = class_labels1[[]]
                    #     # elif args.SingleView == 1:
                    #     #     class_labels0 = class_labels0[[]]
                    #     #     cluster_h0 = cluster_h0[[]]
                    #     # else:
                    #     #     raise NotImplementedError('')
                    is_pair_all.extend(is_pair)
                    feature_vec_cluster.extend(torch.cat([cluster_h0, cluster_h1]).detach().cpu().numpy())

                    group_vec_cluster.extend(torch.concat((torch.zeros(len(cluster_h0), dtype=torch.int),
                                                           torch.ones(len(cluster_h1), dtype=torch.int))).numpy())
                    type_vec_cluster.extend(torch.concat((class_labels0[mask[:, 0] == 1],
                                                          class_labels1[mask[:, 1] == 1])).numpy())
                    feature_vec_classification.extend(torch.cat([h0, h1]).detach().cpu().numpy())

                    if (epoch + 1) == epochs or (epoch + 1) % args.VisualFreq == 0:
                        if torch.sum(torch.logical_not(torch.logical_or(mask[:, 1], mask[:, 0]))):
                            raise NotImplementedError('存在一个pair两个模态都缺失')
                        if args.reFill == 'Copy':
                            if torch.sum(mask[:, 0] == 0):
                                h0[mask[:, 0] == 0] = h1[mask[:, 0] == 0]
                            if torch.sum(mask[:, 1] == 0):
                                h1[mask[:, 1] == 0] = h0[mask[:, 1] == 0]
                        elif args.reFill == 'Center':
                            # raise NotImplementedError('')
                            if self.pred_center_cac is None:
                                pass
                                warnings.warn('self.pred_center_cac == None')
                            else:
                                centors = torch.zeros((len(mask), 2, len(self.pred_center_cac[0]))).cuda()
                                centors[mask[:, 0] == 1, 0] = self.pred_center_cac[
                                    self.pred_cac[:torch.sum(mask[:, 0] == 1)]]
                                centors[mask[:, 1] == 1, 1] = self.pred_center_cac[
                                    self.pred_cac[torch.sum(mask[:, 0] == 1):]]
                                if torch.sum(mask[:, 0] == 0):
                                    h0[mask[:, 0] == 0] = centors[mask[:, 0] == 0, 1]
                                if torch.sum(mask[:, 1] == 0):
                                    h1[mask[:, 1] == 0] = centors[mask[:, 1] == 0, 0]
                        elif args.reFill == 'KnnMapMean':
                            if torch.sum(mask[:, 0] == 0):
                                nearest = get_nearest_k(h1[mask[:, 0] == 0], h1[is_pair], args.reAlignK)
                                h0p = h0[is_pair]
                                h1[mask[:, 0] == 0] = torch.cat([torch.mean(h0p[ns], dim=0) for ns in nearest])
                            if torch.sum(mask[:, 1] == 0):
                                nearest = get_nearest_k(h0[mask[:, 1] == 0], h0[is_pair], args.reAlignK)
                                h1p = h1[is_pair]
                                h1[mask[:, 1] == 0] = torch.cat([torch.mean(h1p[ns], dim=0) for ns in nearest])

                            # raise NotImplementedError('')
                        elif args.reFill == 'KnnMean':
                            # 关联对齐, xi1 不变, xi2替换成离xi1最近的k个view2的点的mean
                            if torch.sum(mask[:, 1] == 0):
                                hs0 = h0[mask[:, 1] == 0]
                                he1 = h1[mask[:, 1] == 1]
                                nearest = get_nearest_k(hs0, he1, args.reAlignK)
                                # nearest = torch.argsort(torch.cdist(hs0.cpu(), he1.cpu()), dim=1)[:, :args.reAlignK]
                                h1[mask[:, 1] == 0] = torch.cat([torch.mean(he1[ns], dim=0) for ns in nearest])
                                # class_labels1[mask[:, 1] == 0] = class_labels1[mask[:, 1] == 1][nearest[:, 0]]
                            if torch.sum(mask[:, 0] == 0):
                                hs1 = h1[mask[:, 0] == 0]
                                he0 = h0[mask[:, 0] == 1]
                                nearest = get_nearest_k(hs1, he0, args.reAlignK)
                                # nearest = torch.argsort(torch.cdist(hs1.cpu(), he0.cpu()), dim=1)[:, :args.reAlignK]
                                h0[mask[:, 0] == 0] = torch.cat([torch.mean(he0[ns], dim=0) for ns in nearest])
                                # class_labels0[mask[:, 0] == 0] = class_labels0[mask[:, 0] == 1][nearest[:, 0]]
                            ###############################################################
                            # 缺失补全, xi2 = mean(离xi1最近的k个view2的点)
                            # fill_num = k
                            # C = euclidean_dist(h0, h1)
                            # row_idx = C.argsort()
                            # col_idx = (C.t()).argsort()
                            # # Mij denotes the flag of i-th sample in view 0 and j-th sample in view 1
                            # M = torch.logical_and((mask[:, 0].repeat(test_num, 1)).t(), mask[:, 1].repeat(test_num, 1))
                            # for i in range(test_num):
                            #     idx0 = col_idx[i, :][
                            #         M[col_idx[i, :], i]]  # idx for view 0 to sort and find the non-missing neighbors
                            #     idx1 = row_idx[i, :][
                            #         M[i, row_idx[i, :]]]  # idx for view 1 to sort and find the non-missing neighbors
                            #     if len(idx1) != 0 and len(idx0) == 0:  # i-th sample in view 1 is missing
                            #         avg_fill = h1[idx1[0:fill_num], :].sum(dim=0) / fill_num
                            #         cnt += (class_labels1[idx1[0:fill_num]] == class_labels1[i]).sum()
                            #         missing_cnt += 1
                            #         recover_out0[i, :] = h0[i, :]
                            #         recover_out1[i, :] = avg_fill  # missing
                            #     elif len(idx0) != 0 and len(idx1) == 0:
                            #         avg_fill = h0[idx0[0:fill_num], :].sum(dim=0) / fill_num
                            #         cnt += (class_labels0[idx0[0:fill_num]] == class_labels0[i]).sum()
                            #         missing_cnt += 1
                            #         recover_out0[i, :] = avg_fill  # missing
                            #         recover_out1[i, :] = h1[i, :]
                            #     elif len(idx0) != 0 and len(idx1) != 0:
                            #         recover_out0[i, :] = h0[i, :]
                            #         recover_out1[i, :] = h1[i, :]
                            #     else:
                            #         raise Exception('error')
                            # if setting == 1:
                            #     align_out0.extend((recover_out0.cpu()).numpy())
                            #     align_out1.extend((recover_out1.cpu()).numpy())
                            #     continue
                            #
                        else:
                            raise NotImplementedError('')

                        to_realign = torch.logical_and(is_pair == 0, torch.logical_and(mask[:, 1], mask[:, 0]))
                        if args.reAlign == 'KnnMean':
                            # 关联对齐, xi1 不变, xi2替换成离xi1最近的k个view2的点的mean
                            if torch.sum(to_realign):
                                ha1 = h1[to_realign]
                                nearest = get_nearest_k(h0[to_realign], ha1, args.reAlignK)
                                # dist = torch.cdist(h0[to_realign].cpu(), ha1.cpu())
                                # nearest = torch.argsort(dist, dim=1)[:, :args.reAlignK]
                                h1[to_realign] = torch.cat([torch.mean(ha1[ns], dim=0) for ns in nearest])
                                # class_labels1[is_pair == 0] = class_labels1[is_pair == 0][nearest[:, 0]]
                        elif args.reAlign == 'Copy':
                            if torch.sum(to_realign):
                                h1[to_realign] = h0[to_realign]
                                # class_labels1[is_pair == 0] = class_labels0[is_pair == 0]
                        elif args.reAlign == 'KnnMapMean':
                            if torch.sum(to_realign):
                                targ_v1 = h1[is_pair]
                                nearest = get_nearest_k(h0[to_realign], h0[is_pair], args.reAlignK)
                                h1[to_realign] = torch.cat([torch.mean(targ_v1[ns], dim=0) for ns in nearest])
                                # class_labels1[is_pair == 0] = ...
                        elif args.reAlign == 'Ignore':
                            pass
                        else:
                            raise NotImplementedError('')

                        if args.Rev:
                            fea0_rec, fea1_rec = self.decode([h1, h0])
                        else:
                            fea0_rec, fea1_rec = self.decode([h0, h1])
                        # if len(fea0_rec[0]) == len(fea1_rec[0]):
                        #     fea_rec = torch.concat([fea0_rec, fea1_rec])
                        #     fea = torch.concat([fea0, fea1])
                        #     mask_c = torch.concat([mask[:, 0], mask[:, 1]])
                        #     if torch.sum(mask_c == 0):
                        #         rnmse_vec[0].extend(
                        #             evaluate.get_rnmse(xs_hat=fea_rec[mask_c == 0], xs=fea[mask_c == 0]).cpu().numpy())
                        #     if torch.sum(mask_c == 1):
                        #         rnmse_vec[1].extend(
                        #             evaluate.get_rnmse(xs_hat=fea_rec[mask_c == 1], xs=fea[mask_c == 1]).cpu().numpy())
                        # else:
                        #     if torch.sum(mask == 0):
                        #         n0_v0 = evaluate.get_rnmse(
                        #             xs_hat=fea0_rec[mask[:, 0] == 0], xs=fea0[mask[:, 0] == 0]).cpu().numpy()
                        #         n0_v1 = evaluate.get_rnmse(
                        #             xs_hat=fea1_rec[mask[:, 1] == 0], xs=fea1[mask[:, 1] == 0]).cpu().numpy()
                        #         rnmse_vec[0].extend(n0_v0)
                        #         rnmse_vec[0].extend(n0_v1)
                        #     if torch.sum(mask == 1):
                        #         n1_v0 = evaluate.get_rnmse(
                        #             xs_hat=fea0_rec[mask[:, 0] == 1], xs=fea0[mask[:, 0] == 1]).cpu().numpy()
                        #         n1_v1 = evaluate.get_rnmse(
                        #             xs_hat=fea1_rec[mask[:, 1] == 1], xs=fea1[mask[:, 1] == 1]).cpu().numpy()
                        #         rnmse_vec[1].extend(n1_v0)
                        #         rnmse_vec[1].extend(n1_v1)

                    g = torch.concat((torch.zeros(len(fea0), device=fea0.device, dtype=torch.int),
                                      torch.ones(len(fea1), device=fea0.device, dtype=torch.int)))

                    h = torch.cat([h0, h1]).detach().cpu().numpy()
                    feature_vec.extend(h)
                    data_vec.extend(torch.cat([fea0, fea1]).detach().cpu().numpy())
                    group_vec.extend(g.cpu().numpy())
                    type_vec.extend(torch.concat((class_labels0, class_labels1)).numpy())

                    inf_data_t = time.time()
            feature_vec = np.array(feature_vec)
            data_vec = np.array(data_vec)
            feature_vec_cluster = np.array(feature_vec_cluster)
            is_pair_all = np.array(is_pair_all)
            feature_vec_classification = np.array(feature_vec_classification)
            group_vec = np.array(group_vec)
            group_vec_cluster = np.array(group_vec_cluster)
            type_vec = np.array(type_vec)
            type_vec_cluster = np.array(type_vec_cluster)
            rnmse_vec[0] = np.array(rnmse_vec[0])
            rnmse_vec[1] = np.array(rnmse_vec[1])
            kmeans_time = TimeOperator.Timer()

            if args.ShowReconstruct:
                if args.dataset == 'MNISTUSPS':
                    dims = [np.product(d.data.shape[1:]) for d in test_dataloader.dataset.datasets]
                    data_list = [np.asarray(it.data, dtype=np.float32) for it in test_dataloader.dataset.datasets]
                    Y = test_dataloader.dataset.datasets[0].targets
                else:
                    dims = [d.shape[1] for d in test_dataloader.dataset.data]
                    data_list = [np.asarray(it, dtype=np.float32) for it in test_dataloader.dataset.data]
                    Y = test_dataloader.dataset.class_labels0

                mask = test_dataloader.dataset.mask
                n_per_cat = 10
                rec0, rec1 = self.decode([
                    torch.from_numpy(feature_vec[group_vec == 0]).cuda(),
                    torch.from_numpy(feature_vec[group_vec == 1]).cuda()])
                rec0 = rec0.detach().cpu().numpy()
                rec1 = rec1.detach().cpu().numpy()
                show_img = np.asarray([])
                inds_map = np.asarray([])
                for v in range(2):
                    col = np.asarray([])
                    inds_map_col = np.asarray([])
                    for y in range(10):
                        inds = np.arange(len(Y))[
                            np.logical_and(np.logical_and(mask[:, v] == 1, mask[:, 1 - v] == 0), Y == y)
                        ]
                        np.random.shuffle(inds)
                        assert len(inds) >= n_per_cat
                        inds = inds[:n_per_cat]
                        raw_imgs = data_list[v][inds]
                        missing_imgs = data_list[1 - v][inds]
                        rec_imgs = [rec0, rec1][v][inds]
                        rec_imgs_miss = [rec0, rec1][1 - v][inds]
                        pack = np.asarray(
                            [raw_imgs, rec_imgs, missing_imgs, rec_imgs_miss]).reshape([-1, n_per_cat, 28, 28])
                        if len(col):
                            col = np.concatenate([col, pack], axis=0)
                        else:
                            col = pack
                        if len(inds_map_col):
                            inds_map_col = np.concatenate([inds_map_col, inds.reshape([1, -1])], axis=0)
                        else:
                            inds_map_col = inds.reshape([1, -1])
                    if len(show_img):
                        show_img = np.concatenate([show_img, col], axis=1)
                    else:
                        show_img = col
                    if len(inds_map):
                        inds_map = np.concatenate([inds_map, inds_map_col], axis=1)
                    else:
                        inds_map = inds_map_col
                plot_heat_map(inds_map, show=True, fig_path='/xlearning/pengxin/Temp/MissingRecIM.svg')
                visualize_image(show_img, show=True, fig_path='/xlearning/pengxin/Temp/MissingRec.svg')
                selected_ind = [
                    [8, 2, 8, 9, 7, 2, 5, 9, 9, 9],
                    [0, 2, 2, 3, 5, 7, 7, 9, 7, 0],
                ]

                # ToMouxin
                inds_to_mouxin = [
                    [im[si] for im, si in zip(inds_map[:, :n_per_cat], selected_ind[0])],
                    [im[si] for im, si in zip(inds_map[:, n_per_cat:], selected_ind[1])],
                ]

                re_dt = np.load(
                    '/xlearning/pengxin/Checkpoints/MultiClustering/RunSets/230105/IMvC_RunSet0114_Ablation_FakeSampleWise/ --QuickConfig X50C50 --dataset MNISTUSPS --loss_sim_contras 0.02 --seed 1998/SampleCache/Np.npz')

                np.savez('/xlearning/pengxin/Temp/MNISTUSPS_show.npz',
                         feature_vec=np.asarray([
                             re_dt['d0_data'][inds_to_mouxin[0]],
                             re_dt['d1_data'][inds_to_mouxin[1]]
                         ]))

                selected_ind_global = np.concatenate(
                    (np.asarray(selected_ind[0]).reshape([-1, 1]),
                     np.asarray(selected_ind[1]).reshape([-1, 1]) + n_per_cat),
                    axis=1
                )

                show_img_final = np.concatenate(
                    [show_img[4 * i:4 * i + 4, selected_ind_global[i]] for i in range(len(selected_ind_global))],
                    axis=1
                )[:, [i * 2 for i in range(10)] + [i * 2 + 1 for i in range(10)]]

                visualize_image(show_img_final, show=True, fig_path='/xlearning/pengxin/Temp/MissingRecFinal.svg')
                return

            def cluster_and_measure(features, types, groups, row_pred=False):
                kst = time.time()
                centroids = torch.from_numpy(kmeans(features, self.class_num))
                if args.ElActivationType in ['Normalize', 'BnNormalize', 'BnReNormalize']:
                    centroids = F.normalize(centroids, dim=1)
                pred_vec = np.argmax(self.soft_ass(torch.from_numpy(features), centroids).numpy(), axis=1)
                pred_adjusted, met = evaluate2(features, pred_vec, types, groups)
                kmeans_time.update(time.time() - kst)
                kmeans_time.show(prefix='kmeans_time')
                if row_pred:
                    return pred_vec, pred_adjusted, met, centroids.cuda()
                else:
                    return pred_adjusted, met, centroids.cuda()

            if not (args.CodeTest and not args.EvalOriMean and not args.EvalOriScoreMean and not args.EvalOriPredMean):
                print('EvalSigel-1')
                pred_vec, pred_adjusted, met, centroids = cluster_and_measure(
                    features=feature_vec_cluster, types=type_vec_cluster, groups=group_vec_cluster, row_pred=True)
                self.pred_cac = pred_vec
                self.pred_center_cac = centroids
            else:
                met = {}
                pred_adjusted = None
                centroids = None
            if args.ShowClustering:
                # sub_sample = args.DrawMax
                # if len(feature_vec) > sub_sample * 2:
                #     ind = np.arange(int(len(feature_vec) // 2))
                #     np.random.shuffle(ind)
                #     ind = ind[:sub_sample]
                #     ind = np.concatenate((ind, ind + int(len(feature_vec) // 2)))
                #     feature_vec = feature_vec[ind]
                #     group_vec = group_vec[ind]
                #     type_vec = type_vec[ind]
                # visualize2(feature_vec=feature_vec, type_vec=type_vec, group_vec=group_vec,
                #            pred_vec=None,
                #            prefix=os.path.join('../', 'Visualization/E{:03d}N{:04d}'.format(epoch, len(type_vec))))
                # visual_image_scatter(
                #     data_vec,
                #     feature_vec,
                #     group_vec,
                #     type_vec,
                # )
                raw_dataset = torchvision.datasets.ImageFolder(
                    'D:/VirtualMachine/Data/caltech-101/101_ObjectCategories',
                    transform=torchvision.transforms.Resize([256, 256])
                )
                # mat = sio.loadmat('D:/VirtualMachine/Data/Caltech101-all.mat')
                # data = mat['X'][0][3:5]
                # label = np.squeeze(mat['Y'])-1

                raw_data_ind = np.ones(len(data_vec), dtype=int) * -1
                class_num = len(np.unique(type_vec))
                class_num_s = len(np.unique(raw_dataset.targets))
                raw_dataset.targets = np.asarray(raw_dataset.targets) - 1
                for t in range(class_num_s):
                    print('{: 4d} {: 4d} {: 4d} {: 4d}'.format(
                        t,
                        np.sum(t == type_vec),
                        np.sum(t == raw_dataset.targets),
                        np.sum(t == 0),

                    ))

                for t in np.unique(type_vec):
                    bank_inds = np.arange(len(raw_dataset.targets))[raw_dataset.targets == t]
                    raw_data_ind[t == type_vec] = np.concatenate([bank_inds, bank_inds])

                # raw_data = raw_dataset[np.asarray(raw_data_ind, dtype=int)]
                raw_data = np.asarray([np.asarray(raw_dataset[it][0]) for it in raw_data_ind])
                np.savez(
                    os.path.join(args.resume.replace('.checkpoint', 'Raw2.npz')),
                    data_vec=raw_data,
                    feature_vec=feature_vec,
                    group_vec=group_vec,
                    type_vec=type_vec,
                    pred_adjusted=pred_adjusted,
                )

                return

            if (epoch + 1) == epochs or (epoch + 1) % args.VisualFreq == 0:
                met_mul2 = {}
                if args.EvalMulti:
                    print('EvalMulti')
                    multi_modality_feature = np.concatenate(
                        [feature_vec[group_vec == view] for view in np.unique(group_vec)],
                        axis=1)
                    _, met_mul, _ = cluster_and_measure(
                        features=multi_modality_feature, types=type_vec[group_vec == 0],
                        groups=group_vec[group_vec == 0])
                    for nv, v in met_mul.items():
                        met_mul2['Multi-' + nv] = v
                if args.EvalMean:
                    print('EvalMean')
                    _, met_mul, _ = cluster_and_measure(
                        features=np.mean(
                            np.asarray([feature_vec[group_vec == view] for view in np.unique(group_vec)]),
                            axis=0
                        ), types=type_vec[group_vec == 0], groups=group_vec[group_vec == 0])
                    for nv, v in met_mul.items():
                        met_mul2['Mean-' + nv] = v
                if args.EvalSingel0:
                    print('EvalSingel0')
                    _, met_mul, _ = cluster_and_measure(features=feature_vec_cluster[group_vec_cluster == 0],
                                                        types=type_vec_cluster[group_vec_cluster == 0],
                                                        groups=group_vec_cluster[group_vec_cluster == 0])
                    for nv, v in met_mul.items():
                        met_mul2['Singel0-' + nv] = v
                if args.EvalSingel1:
                    print('EvalSingel1')
                    _, met_mul, _ = cluster_and_measure(features=feature_vec_cluster[group_vec_cluster == 1],
                                                        types=type_vec_cluster[group_vec_cluster == 1],
                                                        groups=group_vec_cluster[group_vec_cluster == 1] - 1)
                    for nv, v in met_mul.items():
                        met_mul2['Singel1-' + nv] = v
                if args.EvalOriMean:
                    print('EvalOriMean')
                    mean_fea = np.mean(
                        np.asarray([feature_vec[group_vec == view] for view in np.unique(group_vec)]),
                        axis=0
                    )
                    score = self.soft_ass(mean_fea, centroids.cpu().numpy())
                    pred_vec = np.argmax(score, axis=1)
                    _, met_mul = evaluate2(None, pred_vec, type_vec[group_vec == 0], group_vec[group_vec == 0])
                    for nv, v in met_mul.items():
                        met_mul2['OriMean-' + nv] = v
                if args.EvalOriScoreMean:
                    print('EvalOriScoreMean')
                    score = self.soft_ass(torch.from_numpy(feature_vec).cuda(), centroids).cpu().numpy()
                    pred_vec = np.argmax(np.mean(
                        np.asarray([score[group_vec == view] for view in np.unique(group_vec)]),
                        axis=0
                    ), axis=1)
                    _, met_mul = evaluate2(None, pred_vec, type_vec[group_vec == 0], group_vec[group_vec == 0])
                    for nv, v in met_mul.items():
                        met_mul2['OriScoreMean-' + nv] = v
                if args.EvalOriPredMean:
                    print('EvalOriPredMean')
                    pred = torch.softmax(self.soft_ass(torch.from_numpy(feature_vec).cuda(), centroids) / 0.2,
                                         dim=1).cpu().numpy()
                    pred_vec = np.argmax(np.mean(
                        np.asarray([pred[group_vec == view] for view in np.unique(group_vec)]),
                        axis=0
                    ), axis=1)
                    _, met_mul = evaluate2(None, pred_vec, type_vec[group_vec == 0], group_vec[group_vec == 0])
                    for nv, v in met_mul.items():
                        met_mul2['EvalOriPredMean-' + nv] = v
                if args.EvalCla:
                    mv_f = np.asarray([feature_vec[group_vec == view] for view in np.unique(group_vec)])
                    mv_gt = np.asarray([feature_vec_classification[group_vec == view] for view in np.unique(group_vec)])
                    for test_prop in [0.2, 0.5, 0.8]:
                        met_mul2['ClassificationACC{:.01f}'.format(test_prop)] = np.mean(
                            [svm_classify(
                                mv_f,
                                mv_gt,
                                type_vec[group_vec == 0],
                                test_prop=test_prop,
                                C=0.01
                            ) for _ in range(1)]
                        )

                # print('kmeans_time == {}'.format(time.time() - t))
                # met_mul2['rnmse0'] = np.sum(rnmse_vec[0]) / len(rnmse_vec[0])
                # met_mul2['rnmse1'] = np.sum(rnmse_vec[1]) / len(rnmse_vec[1])
                met = {**met, **met_mul2}
            update_log(dic={**args.__dict__, **{'Epoch': epoch}, **met})

            self.train()
            # print('infer_time == {}'.format(time.time() - inf_t))
            timer_infer_time = time.time()

            if (epoch + 1) == epochs or (epoch + 1) % args.VisualFreq == 0 or (args.VisualRandom and epoch == 0):
                np_dir = '../NpPoints/Np{:03d}'.format(epoch)
                DirectoryOperator.FoldOperator(directory=np_dir).make_fold()
                # tn = test_dataloader.dataset.train_sample_num
                # is_pair_all = np.ones(np.sum(group_vec == 0))
                # is_pair_all[test_dataloader.dataset.train_sample_num:] = 0
                # print(test_dataloader.dataset.train_sample_num)
                np.savez(np_dir, feature_vec=feature_vec_cluster, type_vec=type_vec_cluster,
                         group_vec=group_vec_cluster, pred_vec=pred_adjusted, epoch=epoch,
                         mask=test_dataloader.dataset.mask, is_pair_all=is_pair_all)
            if args.CodeTest:
                return

            if epoch == 3:
                evaluate.BestBalance = 0.0
                evaluate.BestEntropy = 0.0
                evaluate.BestFairness = 0.0
                evaluate.BestNmiFair = 0.0

            if False or (
                    epoch >= min(50, args.WarmAll - 5) and np.round(met['acc'], 3) - np.round(best_acc, 3) > 0.0005) \
                    or epoch == args.WarmAll or epoch + 1 == epochs:
                dct = {
                    'epoch'     : epoch,
                    'state_dict': self.state_dict(),
                    'optimizer' : {'optimizer_g': optimizer_g.state_dict()},
                }
                dct = {**dct, 'self_dic': self.__dict__}
                save_checkpoint(dct, epoch=epoch)
                # inference(self, test_dataloader, args)
                best_acc = max(best_acc, met['acc'])
            print('Clu\\g ', end='')
            show_distribution(cluster_vec=pred_adjusted, group_vec=group_vec_cluster, class_num=self.class_num,
                              group_num=self.group_num)
            if not type_detail_shown:
                type_detail_shown = True
                print('Typ\\g ', end='')
                show_distribution(cluster_vec=type_vec_cluster, group_vec=group_vec_cluster, class_num=self.class_num,
                                  group_num=self.group_num)
            show_distribution_ct(type_vec=type_vec_cluster, group_vec=group_vec_cluster, pred_vec=pred_adjusted,
                                 class_num=self.class_num,
                                 group_num=self.group_num)

            if (epoch + 1) % args.VisualFreq == 0 and (args.DrawTSNE or args.DrawUmap):
                if len(feature_vec_cluster) > args.DrawMax:
                    it = np.arange(len(feature_vec_cluster))
                    np.random.shuffle(it)
                    ind = it[:args.DrawMax]
                    feature_vec_visual = feature_vec_cluster[ind]
                    type_vec_visual = type_vec_cluster[ind]
                    group_vec_visual = group_vec_cluster[ind]
                    pred_vec_visual = pred_adjusted[ind]
                else:
                    feature_vec_visual = feature_vec_cluster
                    type_vec_visual = type_vec_cluster
                    group_vec_visual = group_vec_cluster
                    pred_vec_visual = pred_adjusted
                if args.DrawTSNE:
                    visualize(feature_vec=feature_vec_visual, type_vec=type_vec_visual, group_vec=group_vec_visual,
                              pred_vec=pred_vec_visual, prefix='../Visualization/E{:03d}'.format(epoch))
                if args.DrawUmap:
                    UMAP(
                        feature_vec_visual,
                        type_vec_visual,
                        group_vec_visual,
                        pred_vec_visual,
                        self.class_num,
                        self.group_num,
                        args,
                        epoch=epoch,
                    )

            timer_save.update(time.time() - timer_infer_time)

            timer_train_time = time.time()
            self.train()
            confidence_sum = 0.0
            loss_reconstruction_epoch = 0.0
            loss_decenc_epoch = 0.0
            loss_info_balance_epoch = 0.0
            loss_info_fair_epoch = 0.0
            loss_onehot_epoch = 0.0
            loss_consistency_epoch = 0.0
            loss_imsat_epoch = 0.0
            loss_correspondence_epoch = 0.0
            loss_sim_cluster_epoch = 0.0
            loss_sim_instance_epoch = 0.0
            loss_sim_contras_epoch = 0.0
            loss_sim_distrib_epoch = 0.0
            timer_train_data_c = time.time()
            timer_train_data = TimeOperator.Timer()
            for iter, (fea0, fea1, class_labels0, class_labels1, mask, is_pair, idx) in enumerate(train_dataloader):
                timer_train_data.update(time.time() - timer_train_data_c)
                if iter == len(train_dataloader) - 1:
                    timer_train_data.show(prefix='TrainDataTime', total_count=len(train_dataloader),
                                          print_end_time=False)
                if args.SingleView != -1:
                    for v in range(len(mask[0])):
                        if v != args.SingleView:
                            mask[:, v] = 0
                fea0 = fea0[mask[:, 0] == 1].cuda()
                fea1 = fea1[mask[:, 1] == 1].cuda()
                # if args.SingleView != -1:
                #     if args.SingleView == 0:
                #         fea1 = fea1[[]]
                #     elif args.SingleView == 1:
                #         fea0 = fea0[[]]
                #     else:
                #         raise NotImplementedError('')
                g = torch.concat((torch.zeros(len(fea0), device=fea0.device, dtype=torch.int),
                                  torch.ones(len(fea1), device=fea0.device, dtype=torch.int)))
                hp0, hp1 = self.encode([fea0, fea1])
                fea0_, fea1_ = self.decode([hp0, hp1])
                c = self.soft_ass(torch.cat([hp0, hp1]), centroids)
                # exist_mask = torch.cat(mask,dim=0)
                # print('c == {}'.format(c))

                # print('args.BalanceTemperature == {}'.format(args.BalanceTemperature))
                # print('c_ == {}'.format(c_))
                confidence_sum += F.softmax(c / 0.2, dim=1).detach().max(dim=1).values.mean()
                loss = 0
                if args.Reconstruction and epoch < args.ReconstructionEpoch:
                    if args.SingleView == 0:
                        loss_rec = mse_loss(fea0, fea0_)
                    elif args.SingleView == 1:
                        loss_rec = mse_loss(fea1, fea1_)
                    else:
                        loss_rec = (mse_loss(fea0, fea0_) +
                                    mse_loss(fea1, fea1_)) / 2
                    if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                        fe = fea0
                        fe_ = fea0_
                        if len(fe.shape) > 2:
                            fe = fe.view((len(fea0), -1))
                            fe_ = fe_.view((len(fea0_), -1))
                        print('torch.mean(x)=={}, torch.std(x)=={}'.format(torch.mean(fe), torch.std(fe)))
                        # print('torch.min(x)=={}, torch.max(x)=={}'.format(torch.min(fe), torch.max(fe)))
                        print('fe.shape=={}'.format(fe.shape))
                        # print('torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))=={}'.format(
                        #     torch.sqrt(torch.sum(fe ** 2, dim=1, keepdim=False))))
                        # print('torch.sqrt(torch.sum(x ** 2, dim=0, keepdim=True))=={}'.format(
                        #     torch.sqrt(torch.sum(fe ** 2, dim=0, keepdim=False))))
                        print('x[:3]=={}'.format(fe[:3]))
                        print('hp0[:3]=={}'.format(hp0[:3]))
                        print('x_[:3]=={}'.format(fe_[:3]))

                        # print()

                    # loss_rec = 0
                    # print('loss_rec.item() == {}'.format(loss_rec.item()))
                    # print('mse_loss(x[g==0], x_[g==0]).item() == {}'.format(mse_loss(x[g==0], x_[g==0]).item()))
                    # print('mse_loss(x[g==1], x_[g==1]).item() == {}'.format(mse_loss(x[g==1], x_[g==1]).item()))

                    loss = loss_rec * args.Reconstruction
                    loss_reconstruction_epoch += loss_rec.item()
                if epoch >= args.WarmAll:
                    if args.Decenc:
                        if self.group_num <= 2:
                            g_shuffle = 1 - g
                        else:
                            warnings.warn('g_shuffle = g[torch.randperm(c.shape[0])]')
                            g_shuffle = g[torch.randperm(c.shape[0])]
                        lab_g = g_shuffle

                        lg0 = mse_loss(
                            hp0, self.encode(self.decode(hp0.detach(), lab_g[:len(fea0)]), lab_g[:len(fea0)]))
                        lg1 = mse_loss(
                            hp0, self.encode(self.decode(hp0.detach(), lab_g[-len(fea1):]), lab_g[-len(fea1):]))
                        loss_decenc = (lg0 + lg1) / 2
                        loss += loss_decenc * args.Decenc
                        loss_decenc_epoch += loss_decenc.item()
                        raise NotImplementedError('')
                    if args.InfoBalanceLoss or args.InfoFairLoss or True:
                        c_balance = F.softmax(c / args.SoftAssignmentTemperatureBalance, dim=1)

                        O = torch.zeros((self.class_num, self.group_num)).cuda()
                        # E = torch.zeros((self.class_num, self.group_num)).cuda()
                        for b in range(self.group_num):
                            O[:, b] = torch.sum(c_balance[g == b], dim=0)
                            # E[:, b] = (g == b).sum()
                        # E[E <= 0] = torch.min(E[E > 0]) / 10
                        # O[O <= 0] = torch.min(O[O > 0]) / 10
                        if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                            print('c[:3] == \n{}'.format(c[:3]))
                            print('c_balance[:3] == \n{}'.format(c_balance[:3]))
                            O_Hard = torch.zeros((self.class_num, self.group_num)).cuda()
                            for ci, gi in zip(c_balance, g):
                                O_Hard[torch.argmax(ci), gi] += 1
                            print('O_Hard == \n{}'.format(O_Hard))
                            print('O == \n{}'.format(O))
                        pcg = O / torch.sum(O)
                        if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                            print('pcg == \n{}'.format(pcg))

                        if args.InfoBalanceLoss or True:
                            pc = torch.sum(pcg, dim=1, keepdim=False)
                            info_balance_loss = torch.sum(pc * torch.log(pc))
                            loss_info_balance_epoch += info_balance_loss.item() + np.log(self.class_num)
                            if args.InfoBalanceLoss:
                                loss += info_balance_loss * args.InfoBalanceLoss
                            if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                                print('pc == {}'.format(pc))
                                print('pc * torch.log(pc) == {}'.format(pc * torch.log(pc)))

                        if args.InfoFairLoss or True:
                            pc = torch.sum(pcg, dim=1, keepdim=True)
                            pg = torch.sum(pcg, dim=0, keepdim=True)
                            info_fair_loss = torch.sum(pcg * torch.log(pcg / (pc * pg)))
                            loss_info_fair_epoch += info_fair_loss.item()
                            if args.InfoFairLoss:
                                loss += info_fair_loss * args.InfoFairLoss
                    if args.OneHot or True:
                        c_onehot = F.softmax(c / args.SoftAssignmentTemperatureHot, dim=1)
                        loss_onehot = -torch.sum(c_onehot * torch.log(c_onehot)) / float(len(c_onehot))
                        if args.OneHot:
                            loss += loss_onehot * args.OneHot
                        loss_onehot_epoch += loss_onehot.item()
                    if args.loss_self_cons:
                        def get_loss(ob, ev):
                            if args.SelfConsLossType == 'Assignment':
                                return mse_loss(ob, ev)
                            elif args.SelfConsLossType == 'LogAssignment':
                                log_sf = -torch.log_softmax(ob / 0.5, dim=1)
                                if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                                    print('ob[:5, :5] == \n{}'.format(ob[:5, :5]))
                                    print('torch.exp(-log_sf)[:5, :5] == \n{}'.format(torch.exp(-log_sf)[:5, :5]))
                                    print('log_sf[:5, :5] == \n{}'.format(log_sf[:5, :5]))
                                losse = torch.sum(ev * log_sf) / torch.sum(ev)
                                return losse
                            else:
                                raise NotImplementedError('args.SelfConsLossType')

                        hot = torch.eye(self.class_num).cuda()
                        loss_self_cons = 0.
                        # assignment = clusters[idx]
                        warnings.warn('You Are Using Ground Truth.')
                        assignment = torch.cat([class_labels0, class_labels1])
                        c_cons = F.softmax(c / args.SoftAssignmentTemperatureSelfCons, dim=1)
                        for g0 in torch.unique(g):
                            # print('g0 == {}'.format(g0))
                            pm = c_cons[g0 == g]
                            # print('pm.shape == {}'.format(pm.shape))
                            p1m = hot[assignment[g0 == g]]
                            # print('p1m.shape == {}'.format(p1m.shape))
                            # lc = mse_loss(pm @ pm.T, p1m@p1m.T)
                            if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                                print('(pm @ pm.T)[:5, :5] == \n{}'.format((pm @ pm.T)[:5, :5]))
                                print('(p1m@p1m.T)[:5, :5] == \n{}'.format((p1m @ p1m.T)[:5, :5]))

                            loss_self_cons += get_loss(pm @ pm.T, p1m @ p1m.T)
                            # print('loss_self_cons == {}'.format(lc))
                        loss += loss_self_cons * args.loss_self_cons
                        loss_consistency_epoch += loss_self_cons.item()
                        raise NotImplementedError('')
                    if args.loss_sim_cluster:
                        c_cons = F.softmax(c / args.SoftAssignmentTemperatureSelfCons, dim=1)
                        hs0 = c_cons[:len(hp0)]
                        hs1 = c_cons[-len(hp1):]
                        loss_sim = mse_loss(self.soft_ass(hs0, hs0), self.soft_ass(hs1, hs1))
                        loss += loss_sim * args.loss_sim_cluster
                        loss_sim_cluster_epoch += loss_sim.item()
                        raise NotImplementedError('')
                    if args.loss_sim_instance:
                        loss_sim_instance = mse_loss(hp0, hp1)
                        loss += loss_sim_instance * args.loss_sim_instance
                        loss_sim_instance_epoch += loss_sim_instance.item()
                        raise NotImplementedError('')
                    if args.loss_sim_distrib:
                        raise NotImplementedError('')
                        # sij = hp0 @ hp1.T
                        # loss_sim_distrib = (torch.mean(torch.diag(torch.log_softmax(sij, dim=1))) + torch.mean(
                        #     torch.diag(torch.log_softmax(sij, dim=0)))) / -2
                        # loss += loss_sim_distrib * args.loss_sim_distrib
                        # loss_sim_distrib_epoch += loss_sim_distrib.item()
                    if args.loss_imsat:
                        # pass

                        Ip = 1
                        xi = 10
                        nearest_dist = get_dist_release(test_dataloader, os.path.join(
                            # '/xlearning/pengxin/Codes/0412/RunSet-0802_DenoiseSearch_forDist.',
                            '../Imsat',
                            args.dataset + '_' + '10th_neighbor.txt'
                        ))
                        eps_list = nearest_dist[idx].cuda()

                        # eps_list = 0

                        def distance(y0, y1):
                            # compute KL divergence between the outputs of the newtrok
                            def kl(p, q):
                                # compute KL divergence between p and q
                                return torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8))) / float(len(p))

                            return kl(F.softmax(y0, dim=1), F.softmax(y1, dim=1))

                        # print('x.size() == {}'.format(x.size()))
                        # x = None
                        # xsz =

                        # assert len(x.shape) <= 2, 'torch.randn((x.size()[0], x.size()[1])) requires vector'
                        d0 = F.normalize((torch.randn(fea0.size()).view(fea0.size()[0], -1)), p=2, dim=1).view(
                            fea0.size())
                        d1 = F.normalize((torch.randn(fea1.size()).view(fea1.size()[0], -1)), p=2, dim=1).view(
                            fea1.size())
                        c_cons = F.softmax(c / args.SoftAssignmentTemperatureSelfCons, dim=1)

                        for ip in range(Ip):

                            d_var0 = Variable(d0).cuda()
                            d_var0.requires_grad_(True)
                            d_var1 = Variable(d1).cuda()
                            d_var1.requires_grad_(True)

                            y_p = F.softmax(
                                self.soft_ass(torch.cat(self.encode([fea0 + xi * d_var0, fea1 + xi * d_var1])),
                                              centroids) / args.SoftAssignmentTemperatureSelfCons,
                                dim=1)
                            kl_loss = distance(c_cons.detach(), y_p)
                            kl_loss.backward(retain_graph=True)
                            d0 = F.normalize(d_var0.grad, p=2, dim=1)
                            d1 = F.normalize(d_var1.grad, p=2, dim=1)
                        d_var0 = d0.cuda()
                        d_var1 = d1.cuda()
                        eps = args.prop_eps * eps_list
                        eps = eps.view(-1, 1)
                        y_2 = F.softmax(
                            self.soft_ass(torch.cat(self.encode([fea0 + eps * d_var0, fea1 + eps * d_var1])),
                                          centroids) / args.SoftAssignmentTemperatureSelfCons,
                            dim=1)
                        imsat = distance(c_cons.detach(), y_2)
                        loss += imsat * args.loss_imsat
                        loss_imsat_epoch += imsat.item()
                        if (epoch + 1) % args.VisualFreq == 0 and iter == 0:
                            print('eps[:5]=={}\n d_var[:5]=={}\n x[:5]=={}\n eps[:5] * d_var[:5]=={}'.format(
                                eps[:5], d_var0[:5], fea0[:5], eps[:5] * d_var0[:5]))
                            print('c[:5]=={}\n y_2[:5]=={}\n c[:5]-y_2[:5]=={}'.format(
                                c_cons[:5], y_2[:5], c_cons[:5] - y_2[:5]))
                        raise NotImplementedError('')
                    if args.loss_correspondence:
                        c_onehot = F.softmax(c / args.SoftAssignmentTemperatureHot, dim=1)
                        loss_corre = -torch.sum(c_onehot * torch.log(c_onehot + 1e-8)) / float(len(c_onehot))
                        loss += loss_corre * args.loss_correspondence
                        loss_correspondence_epoch += loss_corre.item()
                        raise NotImplementedError('')

                    # def contrastive_loss(self, q, k):
                    #     # normalize
                    #     q = nn.functional.normalize(q, dim=1)
                    #     k = nn.functional.normalize(k, dim=1)
                    #     # gather all targets
                    #     k = concat_all_gather(k)
                    #     # Einstein sum is more intuitive
                    #     logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
                    #     N = logits.shape[0]  # batch size per GPU
                    #     labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
                    #     return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

                if epoch >= args.WarmContras:
                    if args.loss_sim_contras and args.aligned_prop != 0:
                        # hp0_compelet =  hp0[mask[:, 1][mask[:, 0] == 1] == 1]
                        # hp1_compelet =  hp1[mask[:, 0][mask[:, 1] == 1] == 1]
                        sij = self.soft_ass(hp0, hp1) / args.loss_sim_contras_temperature
                        # target = torch.eye(len(is_pair), len(is_pair), device=sij.device)
                        # torch.diag(target) = 0
                        target = torch.diag_embed(is_pair.cuda())
                        target = target[mask[:, 0] == 1][:, mask[:, 1] == 1]

                        # if args.loss_sim_contras_clu:
                        #     c_vec = torch.eye(self.class_num, self.class_num, device=sij.device)
                        #     clu = torch.argmax(c.detach(), dim=1)
                        #     clu0 = c_vec[clu[:len(hp0)]]
                        #     clu1 = c_vec[clu[-len(hp1):]]
                        #     clu_hood = clu0 @ clu1.T
                        #     # clu_hood = c_vec[class_labels0] @ c_vec[class_labels1].T
                        #     # if epoch == args.WarmContras and iter == 0:
                        #     #     warnings.warn('Label leak')
                        #     target = target + args.loss_sim_contras_clu * clu_hood
                        # if args.loss_sim_contras_gau:
                        #     target = target + torch.zeros_like(
                        #         target, device=sij.device).normal_(0, args.loss_sim_contras_gau)
                        def get_loss(sim, tar):
                            is_anchor = torch.sum(tar, dim=1) > 0
                            tar_anchor = tar[is_anchor]
                            if len(tar_anchor) == 0:
                                return 0
                            tar_norm = tar_anchor / torch.sum(tar_anchor, dim=1, keepdim=True)
                            return ce_loss(sim[is_anchor], tar_norm) * 2 * args.loss_sim_contras_temperature

                        # target_a = target / torch.sum(target, dim=1, keepdim=True)
                        # target_b = (target / torch.sum(target, dim=0, keepdim=True)).T
                        loss_sim_contras = get_loss(sim=sij, tar=target) + get_loss(sim=sij.T, tar=target.T)

                        # loss_sim_contras = (
                        #                            torch.mean(torch.diag(torch.log_softmax(sij, dim=1))) +
                        #                            torch.mean(torch.diag(torch.log_softmax(sij, dim=0)))
                        #                    ) / -2
                        loss += loss_sim_contras * args.loss_sim_contras
                        loss_sim_contras_epoch += loss_sim_contras.item()
                        # raise NotImplementedError('')

                optimizer_g.zero_grad()
                loss.backward()
                optimizer_g.step()
                timer_train_data_c = time.time()

            len_train_dataloader = len(train_dataloader)
            confidence_sum /= len_train_dataloader
            loss_reconstruction_epoch /= len_train_dataloader
            loss_decenc_epoch /= len_train_dataloader
            loss_info_balance_epoch /= len_train_dataloader
            loss_info_fair_epoch /= len_train_dataloader
            loss_onehot_epoch /= len_train_dataloader
            loss_consistency_epoch /= len_train_dataloader
            loss_correspondence_epoch /= len_train_dataloader
            loss_imsat_epoch /= len_train_dataloader
            loss_sim_cluster_epoch /= len_train_dataloader
            loss_sim_instance_epoch /= len_train_dataloader
            loss_sim_contras_epoch /= len_train_dataloader
            loss_sim_distrib_epoch /= len_train_dataloader
            print('Epoch [{: 3d}/{: 3d}]'.format(epoch + 1, epochs), end='')
            if loss_reconstruction_epoch != 0:
                print(', Reconstruction:{:04f}'.format(loss_reconstruction_epoch), end='')
            if loss_decenc_epoch != 0:
                print(', Decenc:{:04f}'.format(loss_decenc_epoch), end='')
            if loss_info_balance_epoch != 0:
                print(', InfoBalance:{:04f}'.format(loss_info_balance_epoch), end='')
            if loss_info_fair_epoch != 0:
                print(', InfoFair:{:04f}'.format(loss_info_fair_epoch), end='')
            if loss_consistency_epoch != 0:
                print(', Consistency:{:04f}'.format(loss_consistency_epoch), end='')
            if loss_onehot_epoch != 0:
                print(', OneHot:{:04f}'.format(loss_onehot_epoch), end='')
            if confidence_sum != 0:
                print(', Confidence:{:04f}'.format(confidence_sum), end='')
            if loss_imsat_epoch != 0:
                print(', IMSAT:{:04f}'.format(loss_imsat_epoch), end='')
            if loss_correspondence_epoch != 0:
                print(', Corr:{:04f}'.format(loss_correspondence_epoch), end='')
            if loss_sim_cluster_epoch != 0:
                print(', SimClu:{:04f}'.format(loss_sim_cluster_epoch), end='')
            if loss_sim_instance_epoch != 0:
                print(', SimIns:{:04f}'.format(loss_sim_instance_epoch), end='')
            if loss_sim_contras_epoch != 0:
                print(', SimCon:{:04f}'.format(loss_sim_contras_epoch), end='')
            if loss_sim_distrib_epoch != 0:
                print(', SimDis:{:04f}'.format(loss_sim_distrib_epoch), end='')
            print()
            timer_train.update(time.time() - timer_train_time)
            update_log(dic={**args.__dict__, **{
                'Epoch'                    : epoch + 1,
                'confidence_sum'           : confidence_sum.cpu().numpy(),
                'loss_reconstruction_epoch': loss_reconstruction_epoch,
                'loss_decenc_epoch'        : loss_decenc_epoch,
                'loss_info_balance_epoch'  : loss_info_balance_epoch,
                'loss_info_fair_epoch'     : loss_info_fair_epoch,
                'loss_onehot_epoch'        : loss_onehot_epoch,
                'loss_consistency_epoch'   : loss_consistency_epoch,
                'loss_imsat_epoch'         : loss_imsat_epoch,
                'loss_correspondence_epoch': loss_correspondence_epoch,
                'loss_sim_cluster_epoch'   : loss_sim_cluster_epoch,
                'loss_sim_instance_epoch'  : loss_sim_instance_epoch,
                'loss_sim_contras_epoch'   : loss_sim_contras_epoch,
                'loss_sim_distrib_epoch'   : loss_sim_distrib_epoch,
            }})

            timer_all.update(time.time() - timer_all_time)
            if (epoch + 1) % 1 == 0:
                timer_infer_data.show(prefix='InferDataTime', total_count=len(test_dataloader),
                                      print_end_time=False)
                kmeans_time.show(prefix='kmeans_time')
                timer_save.show(prefix='SaveTime', total_count=args.train_epoch - start_epoch, print_end_time=False)
                timer_train.show(prefix='Train', total_count=args.train_epoch - start_epoch, print_end_time=False)
                timer_all.show(prefix='AllTime', total_count=args.train_epoch - start_epoch)
            print()
            # if args.CodeTest:
            #     return


class VecToPic(nn.Module):
    def __init__(self, depth=1):
        super(VecToPic, self).__init__()
        self.depth = depth

    def __call__(self, x):
        dim = int(np.sqrt(x.shape[1] / self.depth))
        x = x.view((len(x), self.depth, dim, dim))
        return x


class NetCov(Net):
    def __init__(self, args, in_dims, class_num, group_num):
        super(NetCov, self).__init__(args, in_dims, class_num, group_num)

        self.encoder_adaption = nn.ModuleList([
            get_cov([], [], last_layers=[VecToPic()])
            for _ in range(group_num if args.GroupWiseLayer[0] == '1' else 1)])
        self.encoder = nn.ModuleList([
            get_cov([1, 16, 32, 32, 16], [1, 2, 1, 2], with_bn=args.BatchNormType[1] == '1', drop_out=args.Dropout,
                    last_layers=[nn.Flatten()])
            for _ in range(group_num if args.GroupWiseLayer[1] == '1' else 1)])
        self.encoder_linear = nn.ModuleList([
            get_ffn([in_dims[i], 256], with_bn=args.BatchNormType[2] == '1', drop_out=args.Dropout,
                    last_layers=[nn.Linear(256, args.representation_dim)] + self.el_activation_)
            for i in range(group_num if args.GroupWiseLayer[2] == '1' else 1)])
        self.decoder_linear = nn.ModuleList([
            get_ffn([self.dec_in, 256, in_dims[i]], with_bn=args.BatchNormType[3] == '1', drop_out=args.Dropout,
                    last_layers=[VecToPic(depth=16)])
            for i in range(group_num if args.GroupWiseLayer[3] == '1' else 1)])
        self.decoder = nn.ModuleList([
            get_cov([16, 32, 32, 16], [-2, -1, -2], with_bn=args.BatchNormType[4] == '1', drop_out=args.Dropout)
            for _ in range(group_num if args.GroupWiseLayer[4] == '1' else 1)])
        self.decoder_adaption = nn.ModuleList([
            get_cov([], [], last_layers=[nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
                                         nn.Flatten()] + self.final_activation_)
            for _ in range(group_num if args.GroupWiseLayer[5] == '1' else 1)])
