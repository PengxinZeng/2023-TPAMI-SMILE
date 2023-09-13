import os
import sys
import time

import numpy as np
import torch.functional
from matplotlib import pyplot as plt, offsetbox
from matplotlib.ticker import MultipleLocator
from sklearn.manifold import TSNE
# from tsnecuda import TSNE as TsneCuda
from _Utils import DirectoryOperator


def main_calculate_tsne():
    root = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1025_BenchSotaCI22'
    points = ['Epoch149Raw2.npz', ]
    # per_list = [5, 25, 50, 80, 150]
    # per_list = [30, 70, 10, 100]
    per_list = [50]
    [print(it) for it in np.sort(os.listdir(root))]
    folder = 'Checkpoints'
    for np_path in np.sort(os.listdir(root)):
        for np_point in points:
            np_dir = os.path.join(root, np_path, folder, np_point)
            if not os.path.exists(np_dir):
                # print('Not exist {}'.format(np_dir))
                continue
            print('Handling {}'.format(np_dir))
            data_f = np.load(np_dir, allow_pickle=False)

            # feature_vec = data_f['feature_vec']
            # type_vec = data_f['type_vec']
            # group_vec = data_f['group_vec']
            # pred_vec = data_f['pred_vec']
            data = {}
            for k in data_f.keys():
                if k in ['pred_adjusted']:
                    continue
                data[k] = data_f[k]
            DrawMax = 5000
            fea_name = 'feature_vec'
            if len(data[fea_name]) > DrawMax:
                it = np.arange(len(data[fea_name]))
                np.random.shuffle(it)
                ind = it[:DrawMax]
                for k in data.keys():
                    data[k] = data[k][ind]

            for per in per_list:
                np_dir_tsne = os.path.join(root, np_path, folder, "T2P{:03d}".format(per) + np_point)
                if os.path.exists(np_dir_tsne):
                    continue
                data['vis_fea'] = TSNE(perplexity=per).fit_transform(data[fea_name])
                np.savez(
                    np_dir_tsne,
                    **data,
                )


def visual_scatter_multi():
    root = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1025_BenchSotaCI22'
    column_hints = [
        ' --QuickConfig A100 --Rev 1 --dataset NoisyMNIST30000 --normalize_type sample_wise --seed {}',
        # ' --QuickConfig A100 --Rev 1 --dataset NoisyMNIST30000 --normalize_type sample_wise --seed {}',
        ' --QuickConfig C100 --Rev 1 --dataset NoisyMNIST30000 --normalize_type sample_wise --seed {}',
        # ' --QuickConfig C100 --Rev 1 --dataset NoisyMNIST30000 --normalize_type sample_wise --seed {}',
        # ' --QuickConfig A100 --dataset 2view-caltech101-8677sample --seed {}',
        # ' --QuickConfig C100 --dataset 2view-caltech101-8677sample --seed {}',
    ]
    # row_hints = [1999, 1215, 996, 2022, 2023]
    row_hints = [1999, 1215, ]
    # row_hints = [1999]
    per = 50
    x = []
    for row_hint in row_hints:
        x_row_a = []
        x_row_b = []
        for column_ind, column_hint in enumerate( column_hints):
            np_name = 'T1'
            if 'caltech101' in column_hint:
                np_name = 'T2'
            np_name_tail = ''
            if 'caltech101' in column_hint:
                np_name_tail = 'Raw2'
            np_path = os.path.join(root, column_hint.format(row_hint), 'Checkpoints/{}P{:03d}Epoch149{}.npz'.format(
                np_name, per, np_name_tail))
            data = np.load(np_path, allow_pickle=False)
            vis_fea = data['vis_fea']
            data_vec = data['data_vec']
            type_vec = data['type_vec']
            group_vec = data['group_vec']

            # if column_ind%2==0:
            #     res = {'x': data_vec, 'z': vis_fea, 'color': type_vec}
            # else:
            #     res = {'x': data_vec, 'z': vis_fea, 'color': group_vec}
            # x_row_a.append(res)

            res = {'x': data_vec, 'z': vis_fea, 'color': type_vec}
            res_b = {'x': data_vec, 'z': vis_fea, 'color': group_vec}
            x_row_a.append(res)
            x_row_b.append(res_b)
        x.append(x_row_a)
        x.append(x_row_b)

    s = 1
    t = time.time()
    fig_row_num = len(x)
    fig_column_num = len(x[0])
    fig, axs = plt.subplots(fig_row_num, fig_column_num, figsize=(5 * fig_column_num * 2, 5 * fig_row_num * 2 / 1.6))
    # axs = axs.ravel()
    font = 30
    for row_ind, row in enumerate(x):
        for column_ind, data in enumerate(row):
            print("Drawing fig {} {}".format(row_ind, column_ind))
            cur_fig = axs[row_ind, column_ind]

            label_color = data['color']
            color_num = len(np.unique(label_color))
            # if color_num <= 2:
            #     cmap = None
            if color_num <= 10:
                cmap = 'tab10'
            elif color_num <= 20:
                cmap = 'tab20'
            else:
                cmap = 'gist_ncar'

            for c in np.unique(label_color):
                cur_fig.scatter(
                    *data['z'][label_color == c].T,
                    c=label_color[label_color == c],
                    cmap=cmap,
                    s=s,
                    vmax=max(4, np.max(label_color)),
                    vmin=min(0, np.min(label_color)),
                    label=c,
                    alpha=0.7,
                    zorder=2,
                )

            shown_images = np.array([[1.0, 1.0]])  # just something big
            for i in range(data['x'].shape[0]):
                # plot every digit on the embedding
                # show an annotation box for a group of digits
                dist = np.sum((data['z'][i] - shown_images) ** 2, 1)
                if np.min(dist) < 2e1:
                    # don't show points that are too close
                    continue
                if np.min(dist) < 2e1:
                    # don't show points that are too close
                    continue
                shown_images = np.concatenate([shown_images, [data['z'][i]]], axis=0)
                # img = offsetbox.OffsetImage(data_vec[i].reshape([w, h]), cmap=plt.cm.gray_r, )
                if len(data['x'][i].shape) == 1:
                    w = int(np.sqrt(len(data['x'][0])))
                    h = w
                    data_view = data['x'][i].reshape([w, h])
                else:
                    w = data['x'][i].shape[0]
                    data_view = data['x'][i]
                perfect_size = 15
                zoom = perfect_size / w
                img = offsetbox.OffsetImage(data_view, cmap=plt.cm.gray_r, zoom=zoom)
                # img.ti
                imagebox = offsetbox.AnnotationBbox(
                    img,  # [w, h, 3]
                    data['z'][i],
                    pad=0,
                    frameon=False
                )
                imagebox.set(zorder=1)
                cur_fig.add_artist(imagebox)

            cur_fig.set_xticks([])
            cur_fig.set_yticks([])

            # cur_fig.xaxis.set_ticks([])
            if row_ind == 0:
                # if column_ind in [0, 2]:
                #     tt = '\nColored by types'
                # else:
                #     tt = '\ncolored by views'
                tt = ''
                if column_ind in [0]:
                    tt =  r'$100\%$ Unaligned' + tt
                else:
                    tt =  r'$100\%$ Missing'+ tt



                cur_fig.set_title(tt, fontsize=font * 1.2)
            if column_ind == 0:
                lab = 'Seed {:04d}'.format(row_hints[int(row_ind/2)])
                # if row_ind in [0, 1]:
                #     lab = lab + '$\mathcal{L}_{SIL}$'
                # elif row_ind in [2, 3]:
                #     lab = lab + '$\mathcal{L}_{SIL}+\mathcal{L}_{Rec}$'
                # else:
                #     lab = lab + '$\mathcal{L}_{SIL}+\mathcal{L}_{DAR}$'

                if row_ind % 2 == 0:
                    lab = lab + '\nColored by types'

                else:
                    lab = lab + '\ncolored by views'
                cur_fig.set_ylabel(lab, fontsize=font)
            cur_fig.spines['top'].set_visible(False)
            cur_fig.spines['right'].set_visible(False)
            cur_fig.spines['bottom'].set_visible(False)
            cur_fig.spines['left'].set_visible(False)

    plt.tight_layout()
    fig_path = 'D:/Pengxin/Temp/tmp.pdf'
    DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
    plt.savefig(fig_path, transparent=True)
    print('VisualizeScatter finished with in {:.03f} seconds (x.shape == {}).'.format(
        time.time() - t,
        '',
    ))

    # plt.show()
    plt.close()


def visual_image_scatter():
    np_path = os.path.join(
        'D:/VirtualMachine/Codes/230904/SMAIL_RunSet_Visual/ --QuickConfig C100 --VisualFreq 5 --VisualRandom 1 --dataset NoisyMNIST30000 --seed 1999 --train_epoch 100/Checkpoints/Epoch099.npz')
    # np_path_row = os.path.join(root, np_paths[np_names.index(np_tag)], 'NpPoints', np_epoch)

    data = np.load(np_path, allow_pickle=False)
    data_vec = data['data_vec']
    feature_vec = data['feature_vec']
    group_vec = data['group_vec']
    type_vec = data['type_vec']

    # visualize_image(x=[
    #     [it.reshape([28, 28]) for it in data_vec[:10]],
    #     [it.reshape([28, 28]) for it in data_vec[10:20]],
    #     [it.reshape([28, 28]) for it in data_vec[20:30]],
    # ], show=True)

    DrawMax = 3000
    if len(feature_vec) > DrawMax:
        it = np.arange(len(feature_vec))
        np.random.shuffle(it)
        ind = it[:DrawMax]
        feature_vec = feature_vec[ind]
        type_vec = type_vec[ind]
        group_vec = group_vec[ind]
        data_vec = data_vec[ind]
    vis_fea = TSNE(perplexity=50).fit_transform(feature_vec)

    _, ax = plt.subplots(figsize=(5 * 1 * 2, 5 * 1 * 2 / 1.6))

    label_color = np.unique(type_vec)
    color_num = len(np.unique(type_vec))
    # if color_num <= 2:
    #     cmap = None
    if color_num <= 10:
        cmap = 'tab10'
    elif color_num <= 20:
        cmap = 'tab20'
    else:
        cmap = 'gist_ncar'
    for digit in np.unique(type_vec):
        ax.scatter(
            *vis_fea[type_vec == digit].T,
            # marker=f"${digit}$",
            s=0.5,
            # color=plt.cm.Dark2(digit),
            alpha=0.7,
            c=type_vec[type_vec == digit],
            cmap=cmap,
            vmax=max(4, np.max(label_color)),
            vmin=min(0, np.min(label_color)),
            zorder=2,
        )
    w = int(np.sqrt(len(data_vec[0])))
    h = w
    shown_images = np.array([[1.0, 1.0]])  # just something big
    for i in range(data_vec.shape[0]):
        # plot every digit on the embedding
        # show an annotation box for a group of digits
        dist = np.sum((vis_fea[i] - shown_images) ** 2, 1)
        if np.min(dist) < 2e1:
            # don't show points that are too close
            continue
        if np.min(dist) < 2e1:
            # don't show points that are too close
            continue
        shown_images = np.concatenate([shown_images, [vis_fea[i]]], axis=0)
        # img = offsetbox.OffsetImage(data_vec[i].reshape([w, h]), cmap=plt.cm.gray_r, )
        img = offsetbox.OffsetImage(data_vec[i].reshape([w, h]), cmap=plt.cm.gray_r, zoom=0.5)
        # img.ti
        imagebox = offsetbox.AnnotationBbox(
            img,  # [w, h, 3]
            vis_fea[i],
            pad=0,
            frameon=False
        )
        imagebox.set(zorder=1)
        ax.add_artist(imagebox)

    ax.set_title('title')
    ax.axis("off")
    plt.tight_layout()
    plt.savefig('D:/Pengxin/Temp/tmp.pdf')
    plt.show()

    print()
    pass


def visualize_image(x, verbose=0, show=False, fig_path=None):
    """

    :param show:
    :param fig_path:
    :param x:
    (row, line, pic_h, pic_w) or (row, line, pic_h, pic_w, pic_c), pic_c = 1,3,4
    :return:
    """
    x = np.asarray(x)
    if verbose:
        print('img.min() == {}'.format(np.min(x)))
        print('img.max() == {}'.format(np.max(x)))
    x -= np.min(x)
    x /= np.max(x)
    row, line = x.shape[:2]
    w, h = x.shape[1] * x.shape[3] / 90, x.shape[0] * x.shape[2] / 90
    plt.figure(figsize=(w, h))  # w, h
    count = 0
    for rx in x:
        for image in rx:
            count += 1
            plt.subplot(row, line, count)
            plt.imshow(image, cmap='gray', )
            plt.xticks([])
            plt.yticks([])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.1 / h, wspace=0.1 / w)

    if not show and fig_path is None:
        fig_path = '../_fig/fig.jpg'
    if fig_path is not None:
        DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
        plt.savefig(fig_path, transparent=True)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # print(1-0.005**0.1)
    # main_calculate_tsne()
    visual_scatter_multi()
    # visual_image_scatter()

    # main()
