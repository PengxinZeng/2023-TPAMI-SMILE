import os

import numpy as np
import torch

from _Utils.Calculator import get_nearest_k
from _Utils.Visualize import visualize_plot, plot_heat_map


def recall_at_x():
    np_pth = '/xlearning/pengxin/Temp/Python/NpPoints/Np009.npz'
    data_mask_ispair = np.load(np_pth, allow_pickle=False)
    mask = data_mask_ispair['mask']
    is_pair = data_mask_ispair['is_pair_all']
    root = '/xlearning/pengxin/Checkpoints/MultiClustering/RunSets/230105/IMvC_RunSet0114_Ablation_FakeSampleWise/ --QuickConfig X50A50 --dataset NoisyMNIST30000 --loss_sim_contras 0.02 --seed 1998/NpPoints'

    xx = []
    yy = []
    labels = np.round(np.asarray([0.896591937,
                                  0.541940679,
                                  0.525152902,
                                  0.506959305,
                                  ]), decimals=3)
    labels = ['IMvC Epoch{:03d} '.format(ep) + r'$\mathcal{L}_{IsIL}$' + ' = {:5.03f}'.format(it)
              for ep, it in zip([0, 50, 100, 150], labels)]
    # labels = [r'IMvC Epoch{:03d} $\mathcal{L}_{IsIL}$ = {:5.03f}'.format(ep, it) for ep, it in zip([0, 50, 100, 150], labels)]
    # for pf in np.sort(os.listdir(root)):
    for pf in ['Np000.npz', 'Np049.npz', 'Np099.npz', 'Np149.npz', ]:
        if pf.endswith('.npz'):
            np_pth = os.path.join(root, pf)
            print('Handling {}'.format(np_pth))
            # np_pth = '/xlearning/pengxin/Checkpoints/MultiClustering/RunSets/230105/IMvC_RunSet0114_Ablation_FakeSampleWise/ --QuickConfig X50A50 --dataset NoisyMNIST30000 --loss_sim_contras 0.02 --seed 2024/NpPoints/Np149.npz'
            data_f = np.load(np_pth, allow_pickle=False)

            feature_vec = data_f['feature_vec']
            type_vec = data_f['type_vec']
            group_vec = data_f['group_vec']
            pred_vec = data_f['pred_vec']
            epoch = data_f['epoch']

            to_realign = np.logical_and(is_pair == 0, np.logical_and(mask[:, 1], mask[:, 0]))

            h1 = feature_vec[group_vec == 1]
            h0 = feature_vec[group_vec == 0]

            ha1 = h1[to_realign]
            nearest = get_nearest_k(
                torch.from_numpy(h0[to_realign]).cuda(),
                torch.from_numpy(ha1).cuda(),
                np.sum(to_realign)
            ).cpu().numpy()
            gts = np.arange(np.sum(to_realign))
            # h1[to_realign] = torch.cat([torch.mean(ha1[ns], dim=0) for ns in nearest])
            # class_labels1[is_pair == 0] = class_labels1[is_pair == 0][nearest[:, 0]]
            xs = np.linspace(1, 15000, 50, dtype=int)
            y_acu = [0]
            y = []
            for x in xs:
                prop = np.mean([gt in ns[:x] for gt, ns in zip(gts, nearest)])
                # print(prop)
                y.append(prop - y_acu[-1])
                y_acu.append(prop)
            xi = np.round( np.asarray([0] + list(xs)) / np.max(xs) * 100, 0)
            yi = y_acu
            xx.append(xi)
            yy.append(yi)
            # labels.append(pf)

            # visualize_plot([xs], [y], show=True)

    visualize_plot(xx, yy, labels=labels, show=True, fig_path='/xlearning/pengxin/Temp/AlignPlot2.svg', xlim=[-5,55])
    print()


def confusion_matrix():
    np_pth = '/xlearning/pengxin/Temp/Python/NpPoints/Np149.npz'
    data_mask_ispair = np.load(np_pth, allow_pickle=False)
    mask = data_mask_ispair['mask']
    is_pair = data_mask_ispair['is_pair_all']

    np_pth = '/xlearning/pengxin/Checkpoints/MultiClustering/RunSets/230105/IMvC_RunSet0114_Ablation_FakeSampleWise/ --QuickConfig X50A50 --dataset NoisyMNIST30000 --loss_sim_contras 0.02 --seed 2024/NpPoints/Np149.npz'
    data_f = np.load(np_pth, allow_pickle=False)

    feature_vec = data_f['feature_vec']
    type_vec = data_f['type_vec']
    group_vec = data_f['group_vec']
    pred_vec = data_f['pred_vec']
    epoch = data_f['epoch']

    to_realign = np.logical_and(is_pair == 0, np.logical_and(mask[:, 1], mask[:, 0]))
    h1 = feature_vec[group_vec == 1]
    h0 = feature_vec[group_vec == 0]

    ha1 = h1[to_realign]
    nearest = get_nearest_k(
        torch.from_numpy(h0[to_realign]).cuda(),
        torch.from_numpy(ha1).cuda(),
        1
    ).cpu().numpy()

    realign_y = type_vec[group_vec == 1][to_realign][nearest[:, 0]]
    realign_gt = type_vec[group_vec == 0][to_realign]

    conf_y = np.unique(realign_gt)
    conf_x = np.unique(realign_y)
    conf = np.zeros([len(conf_x), len(conf_y)])
    for gt, y in zip(realign_gt, realign_y):
        conf[gt, y] += 1
    for gt in conf_y:
        conf[gt] /= np.sum(gt == realign_gt)
    conf = np.round(conf * 100, decimals=1)
    plot_heat_map(conf, xticks=conf_x, yticks=conf_y, show=True, fig_path='/xlearning/pengxin/Temp/AlignConfusion.svg')

    print()


def reconstruct():
    pass


def main():
    recall_at_x()
    # confusion_matrix()
    # reconstruct()


if __name__ == '__main__':
    main()
