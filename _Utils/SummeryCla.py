import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _Utils.DirectoryOperator import DirectoryOperator
from _Utils.Visualize import visualize_plot

# my_sota_dirs = my_sota_dirs_1027
my_sota_dirs = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1027_ClasGT2'
def get_results_by_dirs(dirs):
    df = pd.DataFrame()
    if isinstance(dirs, str):
        dirs = [dirs]
    for rt in np.sort(dirs):
        for run_root, dirs, files in os.walk(rt):
            if len(run_root) == 0:
                continue
            if run_root[0] == '/':
                run_root = run_root[1:]
            res_csv = os.path.join(run_root, 'log/res.csv')
            if not os.path.exists(res_csv):
                continue
            print('handling {}'.format(res_csv))
            rs = pd.read_csv(res_csv)
            rs.loc[:, 'src'] = run_root
            df = pd.concat([df, rs], ignore_index=True)
    # df = df.fillna(0)
    return df


def get_sota():
    co = ['method', 'dataset', 'aligned_prop', 'complete_prop', 'ClassificationACC0.2', 'ClassificationACC0.5', 'ClassificationACC0.8']
    data = [

        ['KCCA', 'NoisyMNIST30000', 1, 1, 97.20, 97.18, 97.08],
        ['MvCLN', 'NoisyMNIST30000', 0.5, 1, 96.19, 96.18, 96.15],
        # ['SURE', 'NoisyMNIST30000', 0.5, 0, 93.01, 85.40, 85.92],
        # ['SURE', 'NoisyMNIST30000', 0.5, 0.5, 85.04, 67.71, 69.62],


    ]
    df = pd.DataFrame(data, columns=co)
    df.loc[:, 'Name'] = ['{}'.format(m) for m in df.loc[:, 'method']]
    # df.loc[:, 'Name'] = ['{:.1f}/{:.1f}/{}'.format(a, c, m) for a, c, m in zip(
    #     df.loc[:, 'aligned_prop'], df.loc[:, 'complete_prop'], df.loc[:, 'method'])]
    df.set_index('Name', inplace=True)
    return df


def plot():
    # rdir = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1025_BenchSotaCI22'
    # rdir = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1019_InCompelet'
    # [print(rdir + '/' + it) for it in np.sort(os.listdir(rdir))]
    df = get_results_by_dirs(my_sota_dirs)
    sota = get_sota()

    for wm in ['MNISTUSPS', 'NoisyMNIST30000', '2view-caltech101-8677sample', 'AWA-7view-10158sample',
               'cub_googlenet_doc2vec_c10']:
        df_draw = df.copy()
        filter_dic = {'Epoch': 149, 'dataset': wm}
        # filter_dic = {'Epoch': 149}

        group_by_list = [
            'dataset', 'batch_size',
            'aligned_prop', 'complete_prop',
            'CodeTest', 'reFill', 'reAlign', 'reAlignK',
        ]

        for k, v in filter_dic.items():
            df_draw = df_draw.loc[df_draw.loc[:, k] == v]
        if len(group_by_list):
            group_by_list2 = []
            for gn in group_by_list:
                it = df_draw.loc[:, gn]
                if it.isna().sum():
                    it_na = it[- it.isna()]
                    if len(it_na):
                        ddt = it_na.iloc[0]
                    else:
                        ddt = 0
                    df_draw.loc[:, gn] = it.fillna(type(ddt)(0))
                    warnings.warn('Filling nan in {} with {}.'.format(gn, type(ddt)(0)))
                if len(np.unique(it.dropna())) + (1 if it.isna().sum() else 0) > 1:
                    group_by_list2.append(gn)
            group_by_list = group_by_list2
            print(group_by_list)
            dfgd = pd.concat(
                [df_draw[df_draw.loc[:, 'Epoch'] == ep].groupby(group_by_list, as_index=False).mean() for
                 ep in np.unique(df_draw.loc[:, 'Epoch'].values)])
            # dfgd[instance_name] = dfgd.index.values
            df_draw = dfgd
        show_keys = ['aligned_prop', 'complete_prop', 'ClassificationACC0.2', 'ClassificationACC0.5', 'ClassificationACC0.8']
        df_draw = df_draw.loc[:, show_keys]
        df_draw.loc[:, show_keys[2:]] *= 100
        df_draw.loc[:, 'Name'] = 'IMvC'
        df_draw.set_index('Name', inplace=True)

        df_fusion = sota.loc[sota.loc[:, 'dataset'] == wm]
        df_fusion = pd.concat([df_fusion.loc[:, show_keys], df_draw], ignore_index=False)
        # df_fusion = df_draw
        df_fusion.loc[:, show_keys[2:]] = np.round(df_fusion.loc[:, show_keys[2:]], 2)
        df_fusion = df_fusion.sort_values(by=['aligned_prop', 'complete_prop'])
        path = 'D:/VirtualMachine/CheckPoints/MultiClustering/Paper/Table1/Cla_{}.csv'.format(wm)
        DirectoryOperator(path).make_fold()
        df_fusion.to_csv(path)
    print()


def main():
    plot()


if __name__ == '__main__':
    main()
