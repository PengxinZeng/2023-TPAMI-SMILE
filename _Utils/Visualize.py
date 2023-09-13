import os
import sys
import time

import numpy as np
import torch.functional
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.manifold import TSNE
# from tsnecuda import TSNE as TsneCuda
from _Utils import DirectoryOperator

# sys.path.extend(['/xlearning/pengxin/Software/Anaconda/envs/torch171P37/lib'])
# os.execl()

PrintTimer = True


def describ_distribution_image(xs, **kwargs):
    w = len(xs)
    h = 1
    pic = []
    npic = np.min([x.shape[0] for x in xs] + [500])
    for x in xs:
        pic.append(x[:npic])
    pic = np.asarray(pic).reshape((h, w, npic, -1))
    visualize_image(pic, **kwargs)


def describ_distribution(xs, **kwargs):
    mi = np.min([np.min(x) for x in xs])
    ma = np.max([np.max(x) for x in xs])
    line_xs = []
    line_ys = []
    for line in xs:
        x_lim = np.linspace(mi, ma, 100)
        line_xs.append(x_lim)
        ys = [0]
        for x in x_lim:
            y = np.sum(line <= x)
            ys.append(y)
        ys = np.asarray(ys)
        ys = ys[1:] - ys[:-1]
        line_ys.append(ys / np.sum(ys))
    visualize_plot(line_xs, line_ys, **kwargs)
def visualize_plot(x, y, labels=None, show=False, fig_path=None, xlim=None):
    # print('ploting')
    st = time.time()
    plt.figure(figsize=(20, 10))
    if labels is None:
        labels = ['line{:02d}'.format(i) for i in range(len(x))]
    if not show and fig_path is None:
        fig_path = '../_fig/fig.jpg'
    for xi, yi, label in zip(x, y, labels):
        plt.plot(xi, yi, label=label)
    plt.legend(prop={'size': 6})
    plt.grid()
    if xlim is not None:
        plt.xlim((0, 600))

    if fig_path is not None:
        DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
        plt.savefig(fig_path)
    if show:
        plt.show()
    plt.close()
    print('{} lines plotted in {} seconds.'.format(len(x), time.time() - st))


# def visualize_plot(x, y, labels=None, show=False, fig_path=None, xlim=None, ylim=None, spl=None, color=None,
#                    multi_y=True):
#     # print('ploting')
#     st = time.time()
#     # plt.figure()
#     fig, ax = plt.subplots(figsize=(5, 3))
#     # plt.figure(figsize=(5, 3))
#     if labels is None:
#         labels = ['line{:02d}'.format(i) for i in range(len(x))]
#     if not show and fig_path is None:
#         fig_path = '../_fig/fig.jpg'
#     if spl is None:
#         spl_base = ['-', '--', '-.', ':']
#         spl = []
#         for i in range(len(x)):
#             ind = int(i // 10) % len(spl_base)
#             spl.extend(spl_base[ind])
#     # plt.plot(xi, yi, spl[ind], label=label, )
#     for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
#         kwargs = {}
#         if color is not None:
#             kwargs = {'color': color[i]}
#         if multi_y and i:
#             ax2 = ax.twinx()
#             ax2.plot(xi, yi, spl[i], label=label, **kwargs)
#             if color is not None:
#                 ax2.set_ylabel(labels[i],
#                                color=color[i],
#                                fontsize=15)
#         else:
#             ax.plot(xi, yi, spl[i], label=label, **kwargs)
#             if color is not None:
#                 ax.set_ylabel(labels[i],
#                               color=color[i],
#                               fontsize=14, )
#     # plt.legend(prop={'size': 6})
#     # plt.xlabel("x - label")
#     if xlim is not None:
#         # plt.xlim((0, 600))
#         plt.xlim(xlim)
#     if ylim is not None:
#         # plt.xlim((0, 600))
#         plt.ylim(ylim)
#
#     # plt.legend(frameon=False, loc='upper right', ncol=2)
#     ax.grid()
#
#     plt.xticks(x[0], [0] + ['$10^{' + '{}'.format(int(xx - 0.4)) + '}$' for xx in x[0]][1:],
#                # rotation=20
#                )
#     ax.set_xlabel("$\lambda_{SIL}$",
#
#                   fontsize=14, )
#     # plt.xlabel("$\lambda_{SIL}$", fontsize=15)
#     # ax = plt.gca()
#
#     # ax为两条坐标轴的实例
#     # ax.xaxis.set_major_locator(MultipleLocator(2))
#     plt.tight_layout()
#     if fig_path is not None:
#         DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
#         plt.savefig(fig_path, transparent=True)
#     if show:
#         plt.show()
#     plt.close()
#     print('{} lines plotted in {} seconds.'.format(len(x), time.time() - st))


def plot_heat_map(z, xticks=None, yticks=None, xlabel=None, ylabel=None, title=None, show=False, fig_path=None):
    """

    :param z: z[i,j] shown in i-th row, j-th line
    :param xlabel:
    :param ylabel:
    :param show:
    :param fig_path:
    :return:
    """
    left = 0.15
    right = 1
    top = 0.95
    bottom = 0.15
    w, h = z.shape
    plt.figure(figsize=(w / (right - left), h / (top - bottom)))

    # plt.figure(figsize=(w / (right - left), h / (top - bottom)))
    # plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom)

    if xticks is not None:
        plt.xticks(np.arange(len(xticks)), np.round(xticks, 2), rotation=45)
    if yticks is not None:
        plt.yticks(np.arange(len(yticks)), np.round(yticks, 2))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            # plt.text(j, i, accs[i, j].round(2), ha="center", va="center", color="b", fontsize=12,
            #          fontname='Times New Roman')
            plt.text(j, i, z[i, j], ha="center", va="center")

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.imshow(z, interpolation='nearest', aspect='auto')

    plt.colorbar()
    if fig_path is not None:
        DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
        plt.savefig(fig_path, transparent=True)
    if show:
        plt.show()
    plt.close()


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


# Load = False
# Save = not Load
# CountSave = 0

TsneCuda = None


def tsne(x, **kwargs):
    # global CountSave
    t = time.time()
    x = np.asarray(x, dtype=np.float64)
    # if Save:
    #     np.savez('../../np{}'.format(CountSave), x=x)
    #     CountSave += 1
    #     print('save')
    # if Load:
    #     x = np.load('./np.npz')['x']
    # print('len(x) == {}'.format(len(x)))
    # print('start tsne')
    # print(x)
    # print(type(x))
    # print(x.dtype)
    # print(kwargs)
    # y = TSNE(**kwargs, n_jobs=12).fit_transform((x.reshape((len(x), -1))))
    if len(x) < 500000000:
        # print(1)
        y = TSNE(**kwargs, metric='cosine', init='pca', learning_rate='auto').fit_transform((x.reshape((len(x), -1))))
        # y = TSNE(**kwargs).fit_transform((x.reshape((len(x), -1))))
        # y = TSNE(**kwargs).fit_transform((x.reshape((len(x), -1))))
    else:
        # print(2)
        y = TsneCuda(**kwargs, metric='innerproduct').fit_transform((x.reshape((len(x), -1))))
        # y = TsneCuda(**kwargs).fit_transform((x.reshape((len(x), -1))))
        # y = TsneCuda(**kwargs).fit_transform((x.reshape((len(x), -1))))
    if PrintTimer:
        # print(3)
        print('TSNE finished with in {:.03f} seconds (x.shape == {}, y.shape == {}).'.format(
            time.time() - t,
            x.shape,
            y.shape
        ))
    return y


def visualize_scatter(x, label_color=None, label_shape=None, fig_path=None, show=False, DiscriminativeNumber=True,
                      **kwargs):
    """

    :param show:
    :param fig_path:
    :param x: (n_samples, 2)
    :param label_color:
    :param label_shape: list of int
    :return:
    """
    t = time.time()

    if label_color is None:
        label_color = [0] * len(x)
    # print('label_color.shape == {}'.format(label_color.shape))
    # print('label_color == {}'.format(label_color))
    vmin = np.min(label_color)
    vmax = np.max(label_color)
    color_num = len(np.unique(label_color))

    if color_num <= 2:
        cmap = None
    elif color_num <= 10:
        cmap = 'tab10'
    elif color_num <= 20:
        cmap = 'tab20'
    else:
        vmean = (vmax + vmin) / 2
        vd = (vmax - vmin) / 2
        vmax = vmean + vd * 1.1
        vmin = vmean - vd * 1.2
        cmap = 'gist_ncar'
    if DiscriminativeNumber:
        shape_base = 1
    else:
        shape_base = int(np.log10(color_num - 1 + 0.5)) + 1
    if label_shape is None:
        label_shape = [None] * len(x)
    else:
        label_shape = [
            ('${:0' + '{:d}'.format(shape_base) + 'd}$').format(
                (shape % 10) if DiscriminativeNumber else shape) for shape in label_shape
        ]

    fig_size = np.sqrt(len(x)) * 0.10 * shape_base
    w = np.max(x[:, 0]) - np.min(x[:, 0])
    h = np.max(x[:, 1]) - np.min(x[:, 1])
    k = np.sqrt(w / h)
    left = 0.35
    right = 0.25
    w = fig_size * k + left
    h = fig_size / k + right
    plt.figure(figsize=(w, h))
    # plt.figure()
    number_size = 25 * shape_base

    # if InGroup:
    label_shape = np.asarray(label_shape)
    for it in np.unique(label_shape):
        ind = label_shape == it
        plt.scatter(x[ind, 0], x[ind, 1],
                    c=label_color[ind], vmin=vmin, vmax=vmax, cmap=cmap,
                    marker=it, s=number_size,
                    )
    # else:
    #     for i in range(len(x)):
    #         plt.scatter(x[i, 0], x[i, 1],
    #                     c=label_color[i], vmin=vmin, vmax=vmax, cmap='tab20',
    #                     marker=label_shape[i], s=number_size,
    #                     )
    plt.subplots_adjust(left=left / w, right=1, top=1, bottom=right / h)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.grid()
    if fig_path is not None:
        DirectoryOperator.FoldOperator(directory=fig_path).make_fold()
        plt.savefig(fig_path, transparent=True)
    if show:
        plt.show()
    plt.close()
    if PrintTimer:
        print('VisualizeScatter finished with in {:.03f} seconds (x.shape == {}).'.format(
            time.time() - t,
            x.shape,
        ))


def visual_matrix_console(x):
    if len(x.shape) <= 2:
        x = x.reshape((*x.shape, 1))
    base_wid = int(np.log10(np.max(x) + 0.5)) + 1
    head_wid = x.shape[2] * (1 + base_wid)
    head_sep = int(head_wid // 2) + 1
    print('t\\c ', end='')
    for i in range(x.shape[1]):
        print(('{:' + '{}'.format(head_sep) + 'd}').format(i), end=' ' * (head_wid - head_sep))
    print()
    for i, line in enumerate(x):
        print('{:2d}: '.format(i), end='')
        for cl in line:
            sg = True
            for g in cl:
                if sg:
                    sg = False
                else:
                    print(' ', end='')
                if g != 0:
                    # print('base_wid == {}'.format(base_wid))
                    # print('g == {}'.format(g))
                    print(('{:' + str(base_wid) + 'd}').format(g), end='')
                else:
                    print(' ' * base_wid, end='')
            print('|', end='')
        print()


def main():
    UseTsne = True
    # n = 2
    # n = 100
    # 2.089 0.325

    # n = 1000
    # 3.538 0.372

    # n = 4000
    # n = 5000
    # 10.064 0.829

    n = 10000
    # 19.424 1.425
    # 10.347

    # n = 30000
    # 69.801 4.415
    # 10.341

    # n = 100000
    # 10.876 11.9

    # n = 150000
    # 12.365 14.980
    clu = 10
    dim = 10
    x = np.random.random((n, dim if UseTsne else 2))
    # x += t.reshape((-1, 1)) * 0.41

    t = np.arange(n) % clu
    for i, j in enumerate(t):
        x[i, int(j // dim)] += 1
        x[i, int(j % dim)] += 1
    # print(x)
    x = x / np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
    # print(x)
    label_color = t
    label_shape = t
    if UseTsne:
        x = tsne(x)
    ind = len(x) - np.arange(len(x)) - 1
    visualize_scatter(
        x[ind],
        label_color=label_color[ind],
        label_shape=label_shape[ind],
        # fig_path='../c.png',
        show=True,
    )


def main2():
    visualize_image(x=np.random.random((3, 50, 28, 28, 1)), show=True)


def main3():
    def kl(p, q):
        return np.sum([pp * np.log(pp / qq) for pp, qq in zip(p, q)])

    x = np.linspace(0, 1, 51)[1:-1]
    qx = [0.2, 0.8]
    visualize_plot(
        x=[
            x, x, x
        ],
        y=[
            [kl([xx, 1 - xx], qx) for xx in x],
            [kl(qx, [xx, 1 - xx]) for xx in x],
            [(kl(qx, [xx, 1 - xx]) + kl([xx, 1 - xx], qx)) / 2 for xx in x],
        ],
        labels=['1', '2', '3'],
        fig_path='../c.png',
    )


def visualize(feature_vec, type_vec, group_vec, pred_vec, prefix='../Visualization/E{:03d}'.format(0)):
    vis_fea = tsne(feature_vec)
    visualize_scatter(vis_fea,
                      fig_path='{}Type.jpg'.format(prefix),
                      label_color=type_vec,
                      label_shape=type_vec,
                      )
    visualize_scatter(vis_fea,
                      fig_path='{}Cluster.jpg'.format(prefix),
                      label_color=pred_vec,
                      label_shape=type_vec,
                      )
    visualize_scatter(vis_fea,
                      fig_path='{}Group.jpg'.format(prefix),
                      label_color=group_vec,
                      label_shape=type_vec,
                      )
