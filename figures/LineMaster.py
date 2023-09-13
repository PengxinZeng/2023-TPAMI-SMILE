import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

import pandas as pd


def show():
    scr = 'D:/Pengxin/Workplace0805/2023TPAMI_SMAIL/DataInPaper/ExperimentalRes.xlsx'
    dt = pd.read_excel(scr, sheet_name='Fig9')
    dt = dt.loc[np.logical_not(np.asarray(pd.isna(dt.loc[:, 'x_name'])))]
    dt.loc[:, 'line_name'] = ['@'.join([dt.loc[ind, 'line_name_2'], dt.loc[ind, 'line_name_1']]
                                       ) for ind in range(len(dt.loc[:, 'Ind']))]
    dt.loc[:, 'y_name'] = dt.loc[:, 'y_name'].fillna(0)
    fig_row_num = len(np.unique(dt.loc[:, 'y_name']))
    fig_column_num = len(np.unique(dt.loc[:, 'x_name']))
    fig, axs = plt.subplots(fig_row_num, fig_column_num, figsize=(5 * fig_column_num * 2, 5 * fig_row_num * 2 / 1.6))
    axs = axs.ravel()
    fig_order_y = list(np.unique(dt.loc[:, 'y_name']))
    fig_order_x = list(np.unique(dt.loc[:, 'x_name']))
    colors = list(np.unique(dt.loc[:, 'line_name_1']))
    marks = list(np.unique(dt.loc[:, 'line_name_2']))
    mark_set = ['-', '--', ':', '-.']
    x_tic_ind = [ind for ind, it in enumerate(dt.columns) if not isinstance(it, str)]
    x_tic = dt.columns[x_tic_ind]
    font = 30
    for ind in range(len(dt.loc[:, 'Ind'])):
        line = dt.loc[ind]
        fig_ind = fig_order_y.index(line.loc['y_name']) * fig_row_num + fig_order_x.index(line.loc['x_name'])
        cur_fig = axs[fig_ind]

        cur_fig.plot(
            x_tic, line.iloc[x_tic_ind], label=line.loc['line_name'],
            color='C{}'.format(colors.index(line.loc['line_name_1'])),
            linestyle=mark_set[marks.index(line.loc['line_name_2'])]
        )

        # cur_fig.fill_between(u, value[i] - std[i], value[i] + std[i],
        #                      color=color_map[lengends[i]],
        #                      alpha=0.2)
        if not pd.isna(line.loc['subfigure_x_name']):
            cur_fig.set_xlabel(line.loc['subfigure_x_name'], fontsize=font)
        if not pd.isna(line.loc['subfigure_y_name']):
            cur_fig.set_ylabel(line.loc['subfigure_y_name'], fontsize=font)
        if not pd.isna(line.loc['x_name']):
            cur_fig.set_title(line.loc['x_name'], fontsize=font * 1.2)

    for ind, cur_fig in enumerate(axs):
        cur_fig.xaxis.set_major_locator(MultipleLocator(1))
        cur_fig.yaxis.set_major_locator(MultipleLocator(5))
        cur_fig.set_xlim(0, 10)
        cur_fig.set_ylim(45, 100)

        # cur_fig.set_xticks(fontsize=14)
        # cur_fig.set_yticks(fontproperties='Times New Roman', size=14)
        cur_fig.set_xticks([it for it in range(0, 11, 1)], ["{}".format(it) for it in range(0, 11, 1)], fontsize=15)
        cur_fig.set_yticks([it for it in range(45, 105, 5)], ["{}".format(it) for it in range(45, 105, 5)], fontsize=15)

        cur_fig.grid(linestyle='--')

        # cur_fig.legend(frameon=True, )
        if ind == 0:
            cur_fig.legend(frameon=False, loc='lower right', ncol=3, fontsize=11)

    fig.tight_layout()
    plt.savefig('D:/Pengxin/Temp/Fig3ACCNMIARI_UnalignedMissingUnpaired.pdf', transparent=True, )


if __name__ == '__main__':

    show()
