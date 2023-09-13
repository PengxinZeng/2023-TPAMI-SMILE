import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from _Utils import DirectoryOperator
from _Utils.Visualize import PrintTimer


def visualize_scatter(x, fig_path=None, label_color=None, show=False, s=1):
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
    color_num = len(np.unique(label_color))
    # if color_num <= 2:
    #     cmap = None
    if color_num <= 10:
        cmap = 'tab10'
    elif color_num <= 20:
        cmap = 'tab20'
    else:
        cmap = 'gist_ncar'
    plt.figure(figsize=[5,5])
    if len(np.unique(label_color)) < 3:
        ind = np.arange(len(label_color))
        np.random.shuffle(ind)
        for idr in ind:
            plt.scatter(
                x[idr:idr+1, 0], x[idr:idr+1, 1], c=label_color[idr:idr+1], cmap=cmap,
                s=s,
                vmax=max(4, np.max(label_color)),
                vmin=min(4, np.min(label_color)),
                label=label_color[idr]
            )
    else:
        for c in np.unique(label_color):
            plt.scatter(
                x[:, 0][label_color == c], x[:, 1][label_color == c], c=label_color[label_color == c], cmap=cmap,
                s=s,
                vmax=max(4, np.max(label_color)),
                vmin=min(4, np.min(label_color)),
                label=c
            )
    # plt.legend(prop={'size': 6})
    # plt.legend()
    # plt.subplots_adjust(left=left / w, right=1, top=1, bottom=right / h)
    # plt.grid()

    plt.axis('off')
    plt.tight_layout()
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


def visualize2(feature_vec, type_vec, group_vec, pred_vec, prefix, ):
    fv = feature_vec.reshape((len(feature_vec), -1))
    for perplexity in []:# 50
        vis_fea_multi = TSNE(perplexity=perplexity).fit_transform(
            np.concatenate((fv[group_vec == 0], fv[group_vec == 1]), axis=1)
        )
        for s in [5]:
            prefix2 = prefix + 'P{}S{}'.format(perplexity, s)
            visualize_scatter(vis_fea_multi,
                              fig_path='{}Multi.svg'.format(prefix2),
                              label_color=type_vec[group_vec == 0],
                              # label_shape=type_vec,
                              s=s
                              )

    for perplexity in [50]:
        vis_fea = TSNE(perplexity=perplexity).fit_transform(fv)
        for s in [5]: # 5
            prefix2 = prefix + 'P{}S{}'.format(perplexity, s)
            visualize_scatter(vis_fea,
                              fig_path='{}Type.svg'.format(prefix2),
                              label_color=type_vec,
                              # label_shape=type_vec,
                              s=s
                              )
            # visualize_scatter(vis_fea,
            #                   fig_path='{}Cluster.svg'.format(prefix),
            #                   label_color=pred_vec,
            #                   label_shape=type_vec,
            #
            #                   )
            visualize_scatter(vis_fea,
                              fig_path='{}Group.svg'.format(prefix2),
                              label_color=group_vec,
                              # label_shape=type_vec,
                              s=s
                              )
