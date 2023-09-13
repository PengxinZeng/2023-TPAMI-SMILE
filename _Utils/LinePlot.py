import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

import pandas as pd


def show():
    src = 'D:/VirtualMachine/CheckPoints/MultiClustering/Repro/SMAIL/resCompShow.csv'
    df = pd.read_csv(src)
    df = df.fillna(0)

    methods = np.unique(df.loc[:, "Method"])[[2, 5, 0, 3, 1, 4]]
    color_fill_between = ['cornflowerblue', 'gold', 'green', 'salmon', 'grey', 'brown', 'lightskyblue', 'purple',
                          'hotpink', 'goldenrod', 'red']
    # color_fill_between = ['cornflowerblue', 'gold', 'green', 'salmon', 'grey', 'brown', 'lightskyblue', 'purple',
    #                       'hotpink', 'goldenrod', 'red']
    color_map = {
        'IMVC'  : 'C0',
        'SURE'  : 'C1',
        'DCP'   : 'C2',
        'DSIMVC': 'C4',
        'MvCLN' : 'C3',
        'PVC'   : 'C5',

    }
    # color_fill_between_mnist = ['cornflowerblue', 'green', 'salmon', 'brown', 'lightskyblue', 'purple',
    #                             'hotpink', 'goldenrod', 'red']

    aps = np.unique(df.loc[:, "ap"])
    f_types = [
        ['ACC', 'std', 'ACC'],
        ['nmi', 'std.1', 'NMI'],
        ['ari', 'std.2', 'ARI'],
    ]
    # cp_to_draws = [0.5]
    for f_type in f_types:
        cp_to_draws = np.unique(df.loc[:, "cp"])
        for cp_to_draw in cp_to_draws:
            cps = np.ones_like(aps) * cp_to_draw
            # cps = np.unqle(df.loc[:,"cp"])
            value = []
            xs = []
            std = []
            lengends = []

            for method in methods:
                vi = []
                si = []
                xi = []

                for ap, cp in zip(aps, cps):
                    row_id = np.logical_and(
                        df.loc[:, "Method"] == method,
                        df.loc[:, "ap"] == ap,
                    )
                    row_id = np.logical_and(
                        row_id,
                        df.loc[:, "cp"] == cp,
                    )
                    v = df.loc[row_id, f_type[0]]
                    if len(v) and np.min(v):
                        vi.append(v)
                        s = df.loc[row_id, f_type[1]]
                        # if not len(s):
                        #     s = 0
                        si.append(s)
                        xi.append(1 - ap)
                if len(vi):
                    value.append(np.asarray(vi)[:, 0])
                    std.append(np.asarray(si)[:, 0])
                    xs.append(np.asarray(xi))
                    lengends.append(method)
            # method = ["PVC", "IMG", "UEAF", "DAIMC", "PIC", "EERIMVC", "DCCA", "DCCAE", "BMVC", "AE$^2$Nets", "Our"]
            # x = range(2,26,2)
            # y = [15,13,14,5,17,20,25,26,24,22,18,15]
            # 设置图片大小
            # plt.figure(figsize=(20,8),dpi=80) # figsize设置图片大小，dpi设置清晰度
            fig = plt.figure()
            ax1 = fig.add_subplot()

            markers = ['1', 'x', '^', '2', 'o', '*', '+', 'p', 's', 'v']
            # for i in range(11):
            #     u = [(i) / 10 for i in range(len(value[0]))]
            #     if i == 10:
            #         g = plt.plot(u, value[i], label=method[i], color='red')
            #     else:
            #         g = plt.plot(u, value[i], label=method[i],  marker=markers[i], ms=4)
            # Todo: Unpair dot
            # Todo: std

            # color_fill_between = ['cornflowerblue', 'gold', 'green', 'salmon', 'grey', 'brown', 'cyan', 'purple', 'hotpink',
            #                       'goldenrod', 'red']
            num = len(value)
            for i in range(num):

                u = xs[i]
                # u = [(i) / 10 for i in range(len(value[0]))]
                g = plt.plot(u, value[i], label=lengends[i], color=color_map[lengends[i]])

                # if i == 10:
                #     g = plt.plot(u, value[i], label=method[i], color='red')
                # else:
                #     g = plt.plot(u, value[i], label=method[i])
                plt.fill_between(u, value[i] - std[i], value[i] + std[i],
                                 color=color_map[lengends[i]],
                                 alpha=0.2)

            # for i in range(11):
            #     u = [(i) / 10 for i in range(len(value[0]))]
            #
            #     g = plt.plot(u, value[i], label=method[i], color=color_fill_between[i])
            #
            #     # if i == 10:
            #     #     g = plt.plot(u, value[i], label=method[i], color='red')
            #     # else:
            #     #     g = plt.plot(u, value[i], label=method[i])
            #
            #     plt.fill_between(u, value[i] - std[i], value[i] + std[i],
            #                      color=color_fill_between[i],
            #                      alpha=0.2)

            # plt.legend(handles=[li1, li2, li3], labels=[, 'nmi', 'f-measure'])
            # 设置x轴的刻度
            #     plt.xticks(u)
            # 设置y轴的刻度
            #     plt.yticks(range(min(y),max(y)+1)) # 最后一位取不到，所以要加1
            # 保存

            x_major_locator = MultipleLocator(0.1)
            # 把x轴的刻度间隔设置为1，并存在变量里
            y_major_locator = MultipleLocator(10)
            # 把y轴的刻度间隔设置为10，并存在变量里
            ax = plt.gca()

            # ax为两条坐标轴的实例
            ax.xaxis.set_major_locator(x_major_locator)
            # 把x轴的主刻度设置为1的倍数
            ax.yaxis.set_major_locator(y_major_locator)
            # 把y轴的主刻度设置为10的倍数
            # plt.xlim(1, 20)
            # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
            plt.ylim(15, 100)
            plt.grid(linestyle='--')
            plt.legend()
            # plt.legend(loc="upper left", ncol=5, bbox_to_anchor=(1.04, 1),
            #            bbox_transform=plt.gcf().transFigure)

            # ax1.set_xticks(np.arange(0,  20, 2))
            # ax1.set_xticklabels(np.arange(0,  30, 3));

            # ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
            #            mode="expand", borderaxespad=0, ncol=5)

            # ax1.set_xticks(np.arange(1, 10, 1))
            # print(np.arange(0, 0.9, 0.1))
            # x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
            # ax1.set_xticklabels(x);  # use LaTeX formatted labels

            # plt.subplots_adjust(right=0.3)
            x_labels = {
                0  : 'Missing Rate',
                0.5: 'Unpaired Rate',
                1  : 'Unaligned Rate',
            }
            x_label= x_labels[cp_to_draw]
            y_label= f_type[2]
            ax1.set_xlabel(x_label, fontsize=20)
            ax1.set_ylabel(f_type[2], fontsize=20)
            fig.subplots_adjust(top=0.957,
                                bottom=0.151,
                                left=0.133,
                                right=0.977, )
            # plt.show()
            plt.savefig('D:/VirtualMachine/Temp/fig/{}_{}.svg'.format(y_label,x_label ),transparent=True, )
            print()
        print()


if __name__ == '__main__':

    show()
