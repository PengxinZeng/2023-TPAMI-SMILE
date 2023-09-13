import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from _MainLauncher import Queue
from _Utils.Calculator import get_nearest_k
from _Utils.Scatter import visualize_scatter, visualize2
from _Utils.Visualize import plot_heat_map, visualize_plot


def confusion_matrix():
    np_pth = 'D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SMAIL/RaAlign/Base.npz'
    data_mask_ispair = np.load(np_pth, allow_pickle=False)
    mask = data_mask_ispair['mask']

    use_mx = False
    for ep in [59, 89, 119, 149]:
        # for ep in [59, 89, 119]:

        for tem in [50] if use_mx else [0.5]:
            if not use_mx:
                # 50A
                # np_pth = 'D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SMAIL/RaAlign/Np{:03d}.npz'.format(ep)
                # is_pair = data_mask_ispair['is_pair_all']
                # to_realign = np.logical_and(is_pair == 0, np.logical_and(mask[:, 1], mask[:, 0]))

                # 100A
                np_pth = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1025_BenchSotaCI22/ --QuickConfig A100 --Rev 1 --dataset NoisyMNIST30000 --normalize_type sample_wise --seed 9116/NpPoints/Np{:03d}.npz'.format(
                    ep)
                to_realign = np.ones_like(data_mask_ispair['is_pair_all'], dtype=bool)

                data_f = np.load(np_pth, allow_pickle=False)

                feature_vec = data_f['feature_vec']
                type_vec = data_f['type_vec']
                group_vec = data_f['group_vec']
                pred_vec = data_f['pred_vec']
                epoch = data_f['epoch']
                type_vec = type_vec[group_vec == 0]

                h1 = feature_vec[group_vec == 1]
                h0 = feature_vec[group_vec == 0]
                ha0 = h0[to_realign]
                ha1 = h1[to_realign]
                gt = type_vec[to_realign]
                # ind = np.argsort(gt)
                # ha0 = ha0[ind]
                # ha1 = ha1[ind]
                # gt = gt[ind]

                # ha0[np.sum(ha0 ** 2, axis=1) == 0] = 1e-8
                # ha1[np.sum(ha1 ** 2, axis=1) == 0] = 1e-8
                # ha0 = ha0 / np.sqrt(np.sum(ha0 ** 2, axis=1, keepdims=True))
                # ha1 = ha1 / np.sqrt(np.sum(ha1 ** 2, axis=1, keepdims=True))
                sim = torch.from_numpy(ha0 @ ha1.T)

                # ind_score = np.zeros(len(ha0))
                # for y in np.unique(gt):
                #     cy = conf[y == gt][:, y == gt]
                #     cy = (cy + cy.T) / 2
                #     ind_score[y == gt] = np.mean(cy, axis=1) - 10 * y
                # ind = np.argsort(ind_score)[::-1]

                # tem = 0.2
            else:
                # np_pth = 'D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SURE/NoisyMNIST50align.npz'
                # data_f = np.load(np_pth, allow_pickle=False)
                np_pth = 'D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SURE/NoisyMNIST50align.npz'
                data_f = np.load(np_pth, allow_pickle=False)
                feature_vec = data_f['feature_vec']
                is_pair = data_f['is_pair']
                h0 = feature_vec[:30000]
                h1 = feature_vec[30000:]
                type_vec0 = data_f['type_vec'][:30000]
                type_vec1 = data_f['type_vec'][30000:]

                to_realign = is_pair == 0
                epoch = 0
                ha0 = h0[to_realign]
                ha1 = h1[to_realign]
                gt0 = type_vec0[to_realign]
                gt1 = type_vec1[to_realign]

                ind = np.argsort(gt0)
                ha0 = ha0[ind]
                gt0 = gt0[ind]

                ind = np.argsort(gt1)
                ha1 = ha1[ind]
                gt1 = gt1[ind]
                assert np.sum(gt0 != gt1) == 0
                gt = gt1
                sim = -torch.cdist(torch.from_numpy(ha0), torch.from_numpy(ha1))
                # tem = 1
            figp = 'D:/Pengxin/Temp/AlignConfusionE{:03d}T{:05.01f}{:}.svg'.format(epoch, tem, 'M' if use_mx else '')
            print(figp)

            # feature_vec = torch.from_numpy(feature_vec)
            # # fv = torch.cat([fv[:30000], fv[30000:]], dim=1)
            # prefix = 'D:/Pengxin/Temp/mxv/'
            # group_vec = np.concatenate([np.zeros(30000, dtype=int), np.ones(30000, dtype=int)])
            # type_vec = data_f['type_vec']
            # sub_sample = 3000
            # if len(feature_vec) > sub_sample * 2:
            #     ind = np.arange(int(len(feature_vec) // 2))
            #     np.random.shuffle(ind)
            #     ind = ind[:sub_sample]
            #     ind += 30000
            #     # ind = np.concatenate((ind, ind + int(len(feature_vec) // 2)))
            #     feature_vec = feature_vec[ind]
            #     group_vec = group_vec[ind]
            #     type_vec = type_vec[ind]
            # visualize2(feature_vec, type_vec,
            #            group_vec,
            #            None,
            #            prefix,
            #            )
            sas01 = torch.softmax(sim / tem, dim=1)
            sas10 = torch.softmax(sim / tem, dim=0)
            sas = ((sas01 + sas10) / 2).numpy()

            realign_y = gt[torch.argmax(sas01, dim=1)]
            realign_gt = gt
            acc = np.round(np.sum(realign_y == realign_gt) / len(realign_y) * 100, decimals=1)
            print(acc)
            continue
            # break
            # ind_score = np.zeros(len(ha0))
            # for y in np.unique(gt):
            #     cy = sas[y == gt][:, y == gt]
            #     ind_score[y == gt] = np.mean(cy, axis=1) - 10 * y
            # ind = np.argsort(ind_score)[::-1]
            # conf = sas[ind][:, ind]
            ind_score = np.zeros(len(ha0))
            for y in np.unique(gt):
                cy = sas[y == gt][:, y == gt]
                ind_score[y == gt] = np.mean(cy, axis=1) - 10 * y
            ind = np.argsort(ind_score)[::-1]
            sas = sas[ind]

            ind_score = np.zeros(len(ha0))
            for y in np.unique(gt):
                cy = sas[y == gt][:, y == gt]
                ind_score[y == gt] = np.mean(cy, axis=0) - 10 * y
            ind = np.argsort(ind_score)[::-1]
            conf = sas[:, ind]

            # conf = np.round(conf, decimals=2)

            z = conf
            # plot_heat_map(conf, show=True,
            #               fig_path='/D:/VirtualMachine/CheckPoints/MultiClustering/1014/FigNP/AlignConfusion.svg')
            plt.figure()
            # for i in range(z.shape[0]):
            #     for j in range(z.shape[1]):
            #         # plt.text(j, i, accs[i, j].round(2), ha="center", va="center", color="b", fontsize=12,
            #         #          fontname='Times New Roman')
            #         plt.text(j, i, z[i, j], ha="center", va="center")
            plt.imshow(z, interpolation='nearest', aspect='auto')
            # plt.colorbar()
            plt.axis('off')
            # plt.subplots_adjust(top=0.957, bottom=0.081, left=0.108, right=0.977, hspace=0.2, wspace=0.2)
            # plt.show()
            plt.tight_layout()
            plt.savefig(
                figp,
                transparent=True)
            print()
        if use_mx:
            break


# def scatter():
#     mx = False
#     if mx:
#         np_pth = 'D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SURE/NoisyMNISTBoth.npz'
#         data_f = np.load(np_pth, allow_pickle=False)
#         feature_vec = data_f['feature_vec']
#         type_vec = data_f['type_vec']
#         group_vec = np.concatenate((np.zeros(30000), np.ones(30000)))
#     else:
#         np_pth = 'D:/Pengxin/Workplace/Codes/MultimodalityClustering/Np099.npz'
#         data_f = np.load(np_pth, allow_pickle=False)
#         feature_vec = data_f['feature_vec']
#         type_vec = data_f['type_vec']
#         group_vec = data_f['group_vec']
#
#     sub_sample = 3000
#     if len(feature_vec) > sub_sample * 2:
#         ind = np.arange(int(len(feature_vec) // 2))
#         np.random.shuffle(ind)
#         ind = ind[:sub_sample]
#         ind = np.concatenate((ind, ind + 30000))
#         feature_vec = feature_vec[ind]
#         group_vec = group_vec[ind]
#         type_vec = type_vec[ind]
#     # fv = feature_vec.reshape((len(feature_vec), -1))
#     prefix = 'D:/Pengxin/Temp/mxF/'
#     visualize2(feature_vec, type_vec, group_vec, None, prefix, )
#
#     # for perplexity in [15,30]:
#     #     prefix = 'D:/Pengxin/Temp/mxF/P{}N{}'.format(perplexity, sub_sample)
#     #     vis_fea_multi = TSNE(perplexity=perplexity).fit_transform(
#     #         np.concatenate((fv[group_vec == 0], fv[group_vec == 1]), axis=1)
#     #     )
#     #     visualize_scatter(vis_fea_multi,
#     #                       fig_path='{}Multi.{}'.format(prefix, post),
#     #                       label_color=type_vec[group_vec == 0],
#     #                       )
#     #
#     #     vis_fea = TSNE(perplexity=perplexity).fit_transform(fv)
#     #
#     #     visualize_scatter(vis_fea,
#     #                       fig_path='{}Type.{}'.format(prefix, post),
#     #                       label_color=type_vec,
#     #
#     #                       )
#     #     # visualize_scatter(vis_fea,
#     #     #                   fig_path='{}Cluster.svg'.format(prefix),
#     #     #                   label_color=pred_vec,
#     #     #                   label_shape=type_vec,
#     #     #
#     #     #                   )
#     #     visualize_scatter(vis_fea,
#     #                       fig_path='{}Group.{}'.format(prefix, post),
#     #                       label_color=group_vec,
#     #
#     #                       )


def plot_nrmse():
    rs = pd.read_csv('D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SMAIL/Recover/_Table_NRMSE_AVG_FuEp.csv')
    # rs = pd.read_csv('/xlearning/pengxin/Codes/230318/IMvC_RunSet0405/_Table_NRMSE_AVG_FuEp.csv')
    xx = []
    yy = []
    labels = []
    spls = []
    colors = []
    for i, rec in enumerate(np.unique(rs.loc[:, 'Reconstruction'])):
        if rec > 1:
            continue
        label = '$\mathcal{L}_{IsIL}@\lambda_{IsIL}$' + '={}'.format(rec)
        rd = rs.loc[rs.loc[:, 'Reconstruction'] == rec]
        x = rd.loc[:, 'Epoch']
        y = rd.loc[:, 'loss_reconstruction_epoch']
        xx.append(x)
        yy.append(y)
        labels.append(label)
        spls.append('-')
        colors.append('C{}'.format(i))
    for i, rec in enumerate(np.unique(rs.loc[:, 'Reconstruction'])):
        if rec > 1:
            continue
        label = '$NRMSE@\lambda_{IsIL}$' + '={}'.format(rec)
        rd = rs.loc[rs.loc[:, 'Reconstruction'] == rec]
        ind = pd.isna(rd.loc[:, 'rnmse0']).values != True
        rd2 = rd.loc[ind]
        ind = rd2.loc[:, 'Epoch'].values > 0
        rd2 = rd2.loc[ind]
        x = rd2.loc[:, 'Epoch']
        y = rd2.loc[:, 'rnmse0']
        xx.append(x)
        yy.append(y)
        labels.append(label)
        spls.append('--')
        colors.append('C{}'.format(i))
    # np_pth = 'D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SURE/nrmse_list.npz'
    # data_f = np.load(np_pth, allow_pickle=True)
    # data0 = np.asarray([it.cpu().numpy() for it in  data_f['data0']])
    # data1 = np.asarray([it.cpu().numpy() for it in  data_f['data1']])
    # y = (data1 + data0)/2
    # x = np.arange(len(y))
    # xx.append(x)
    # yy.append(y)
    # labels.append('SURE')
    # spls.append('-.')
    # colors.append('C{}'.format(6))
    visualize_plot(xx, yy, labels=labels, show=True,
                   xlim=[0, 20],
                   ylim=[0, 0.5],
                   spl=spls,
                   color=colors,
                   fig_path='D:/Pengxin/Temp/Fig5NRMSE.pdf'
                   )


def plot_nrmse_RI():
    rs = pd.read_csv('D:/VirtualMachine/Codes/230318/IMvC_RunSet0409_3/_Table_NRMSE_AVG.csv')
    # rs = pd.read_csv('/xlearning/pengxin/Codes/230318/IMvC_RunSet0405/_Table_NRMSE_AVG_FuEp.csv')
    xx = []
    yy = []
    labels = []
    spls = []
    colors = []

    # ind = rs.loc[:, 'Epoch'].values <= 54
    # rs = rs.loc[ind]

    for i, ibl in enumerate(np.unique(rs.loc[:, 'InfoBalanceLoss'])):
        label = '$\mathcal{L}_{SIL}$' + '={}'.format(ibl)
        rd = rs.loc[rs.loc[:, 'InfoBalanceLoss'] == ibl]
        x = rd.loc[:, 'Lsil']
        y = rd.loc[:, 'rnmse0']
        xx.append(x)
        yy.append(y)
        labels.append(label)
        spls.append('-')
        colors.append('C{}'.format(i))
    # for i, rec in enumerate(np.unique(rs.loc[:, 'InfoBalanceLoss'])):
    #     if rec > 1:
    #         continue
    #     label = '$NRMSE@\lambda_{IsIL}$' + '={}'.format(rec)
    #     rd = rs.loc[rs.loc[:, 'Reconstruction'] == rec]
    #     ind = pd.isna(rd.loc[:, 'rnmse0']).values != True
    #     rd2 = rd.loc[ind]
    #     ind = rd2.loc[:, 'Epoch'].values > 0
    #     rd2 = rd2.loc[ind]
    #     x = rd2.loc[:, 'Epoch']
    #     y = rd2.loc[:, 'rnmse0']
    #     xx.append(x)
    #     yy.append(y)
    #     labels.append(label)
    #     spls.append('--')
    #     colors.append('C{}'.format(i))
    # np_pth = 'D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SURE/nrmse_list.npz'
    # data_f = np.load(np_pth, allow_pickle=True)
    # data0 = np.asarray([it.cpu().numpy() for it in  data_f['data0']])
    # data1 = np.asarray([it.cpu().numpy() for it in  data_f['data1']])
    # y = (data1 + data0)/2
    # x = np.arange(len(y))
    # xx.append(x)
    # yy.append(y)
    # labels.append('SURE')
    # spls.append('-.')
    # colors.append('C{}'.format(6))
    # plt.scatter
    visualize_plot(xx, yy, labels=labels, show=True,
                   # xlim=[0, 20],
                   # ylim=[0, 0.5],
                   spl=spls,
                   color=colors,
                   fig_path='D:/Pengxin/Temp/Fig5NRMSE.pdf'
                   )


def plot_nrmse_RI_Lm():
    rs = pd.read_csv('D:/VirtualMachine/CheckPoints/MultiClustering/230318/IMvC_RunSet0409_3/_Table_NRMSE_AVG.csv')
    # rs = pd.read_csv('/xlearning/pengxin/Codes/230318/IMvC_RunSet0405/_Table_NRMSE_AVG_FuEp.csv')
    xx = []
    yy = []
    labels = []
    spls = []
    colors = []

    # ind = rs.loc[:, 'Epoch'].values == 55
    # rs = rs.loc[ind]
    ind = rs.loc[:, 'InfoBalanceLoss'].values != 1e-3
    rs = rs.loc[ind]
    ind = rs.loc[:, 'InfoBalanceLoss'].values != 4
    rs = rs.loc[ind]

    for i, ep in enumerate(np.arange(54, 55)):
        # if ibl == 0:
        #     continue
        ind = rs.loc[:, 'Epoch'].values == ep
        rs2 = rs.loc[ind]
        label = '$RNMSE$'
        # label = '$RNMSE$@Epoch' + '={}'.format(ep)
        rd = rs2.loc[rs2.loc[:, 'Epoch'] == ep]
        x = np.log10(rd.loc[:, 'InfoBalanceLoss'] / 4 + 1e-5)
        y = rd.loc[:, 'rnmse0']
        xx.append(x)
        yy.append(y)
        labels.append(label)
        spls.append('-')
        colors.append('C{}'.format(i))
    for i, ep in enumerate(np.arange(54, 55)):
        # if ibl == 0:
        #     continue
        ind = rs.loc[:, 'Epoch'].values == ep
        rs2 = rs.loc[ind]
        label = '$\mathcal{L}_{SIL}$'
        # label = '$\mathcal{L}_{SIL}$@Epoch' + '={}'.format(ep)
        rd = rs2.loc[rs2.loc[:, 'Epoch'] == ep]

        x = np.log10(rd.loc[:, 'InfoBalanceLoss'] / 4 + 1e-5)
        y = rd.loc[:, 'Lsil']
        xx.append(x)
        yy.append(y)
        labels.append(label)
        spls.append('-')
        colors.append('C{}'.format(i + 1))
    visualize_plot(xx, yy, labels=labels, show=True,
                   # xlim=[0, 20],
                   # ylim=[0, 0.5],
                   spl=spls,
                   color=colors,
                   fig_path='D:/Pengxin/Temp/Fig5NRMSELSIL_lamdaSIL.pdf'
                   )


def main():
    # confusion_matrix()
    # scatter()
    plot_nrmse_RI_Lm()


if __name__ == '__main__':
    main()
    # Queue.dequeue()

# scp -r pengxin@192.168.49.47:"\xlearning\pengxin\Checkpoints\MultiClustering\RunSets\230105\IMvC_RunSet0114_Ablation_FakeSampleWise\ --QuickConfig X50A50 --dataset NoisyMNIST30000 --loss_sim_contras 0.02 --seed 2024\NpPoints" "."
# scp -r pengxin@192.168.49.47:"/xlearning/pengxin/Checkpoints/MultiClustering/RunSets/230105/IMvC_RunSet0114_Ablation_FakeSampleWise/ --QuickConfig X50A50 --dataset NoisyMNIST30000 --loss_sim_contras 0.02 --seed 2024/NpPoints" "."
