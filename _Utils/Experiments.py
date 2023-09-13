import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _Utils.Visualize import visualize_plot


def load_results(root):
    """
    read results and save
    """
    # cache_dir = os.path.join(root, '_result_cache')
    df = pd.DataFrame()
    for rt, dirs, files in os.walk(root):
        for run_root in np.sort(dirs):
            # if run_root.startswith('_QueueLog'):
            #     continue
            # if not os.path.exists(os.path.join(rt, run_root, 'Train.txt')):
            #     continue
            res_csv = os.path.join(rt, run_root, 'log/res.csv')
            if not os.path.exists(res_csv):
                continue
            print('handling {}'.format(res_csv))
            rs = pd.read_csv(res_csv)
            rs.loc[:, 'src'] = os.path.join(rt, run_root)
            df = pd.concat([df, rs], ignore_index=True)
    # df = df.fillna(0)
    return df


def plot():
    # root = 'D:/VirtualMachine/Codes/230318/IMvC_RunSet0409_3'
    # root = 'D:/VirtualMachine/CheckPoints/MultiClustering/1014/RunSet1025_BenchSotaCI22'
    root = 'D:/VirtualMachine/Checkpoints/MultiClustering/'

    # root = '/xlearning/pengxin/Codes/230105/IMvC_RunSet0129_AblationBeta3'
    # root = '/xlearning/pengxin/Codes/230105/IMvC_RunSet0118_C100_2'
    # root = '/xlearning/pengxin/Codes/230318/IMvC_RunSet0429_4'


    df = load_results(root=root)
    # df['SeIn'] = df.loc[:, 'Desc'] // 5
    instance_name = 'src'
    # instance_name = 'Desc'
    mets_to_draw = [
        'acc', 'nmi', 'NmiFair', 'Fmeasure',
        'Mean-acc', 'Multi-acc', 'Multi-nmi', 'Multi-ari', 'Singel0-acc', 'Singel1-acc',
        'loss_reconstruction_epoch',
        'loss_info_balance_epoch', 'balance', 'loss_info_fair_epoch', 'loss_onehot_epoch', 'confidence_sum',
        'loss_sim_cluster_epoch', 'loss_sim_instance_epoch', 'loss_sim_contras_epoch', 'loss_sim_distrib_epoch',
        'loss_imsat_epoch', 'loss_consistency_epoch',
        'ClassificationACC0.2', 'ClassificationACC0.5', 'ClassificationACC0.8',
        'rnmse0','rnmse1',
    ]
    # for wm in ['dim_wise', 'sample_wise', 'rescale_dim_wise', 'rescale_sample_wise', 'None']:
    # for wm in [0.01]:
    # for wm in ['']:
    # for wm in ['cub_googlenet_doc2vec_c10']:
    # for wm in ['2view-caltech101-8677sample', 'AWA-7view-10158sample', 'NoisyMNIST30000','MNISTUSPS', 'cub_googlenet_doc2vec_c10']:
    # for wm in [False]:
    # for wm in [[0,1]]:
    # for wm in [False]:
    # for wm in [[0.5, 1], [1, 1]]:
    # for wm in ['AWA-7view-10158sample']:
    # for wm in ['2view-caltech101-8677sample', 'AWA-7view-10158sample', 'NoisyMNIST30000','MNISTUSPS', 'cub_googlenet_doc2vec_c10']:
    for wm in [""]:
        df_draw = df.copy()
        filter_dic = {}
        # filter_dic = {'batch_size': 64}
        # filter_dic = {'dataset': 'NoisyMNIST30000', 'loss_sim_contras': 0.5, 'CodeTest': wm, 'Epoch': 69}
        # filter_dic = {'dataset': wm}
        # filter_dic = {'aligned_prop': 0.5, 'complete_prop': 0}
        # filter_dic = {'Epoch': 149, 'dataset': 'MNISTUSPS', 'aligned_prop': wm[0], 'complete_prop': wm[1]}
        # filter_dic = {'Epoch': 149}
        filter_dic = {  'Reconstruction': 0, 'Epoch': 149}
        # filter_dic = {  'complete_prop': 0, 'InfoBalanceLoss': 0.04}
        # filter_dic = {'dataset': 'cub_googlenet_doc2vec_c10'}
        # filter_dic = {'dataset': wm}
        # filter_dic = {'Epoch': 149}
        # filter_dic = {'SeIn': 7}

        # filter_dic = {'withTanh': 0, 'ActivationType': 'None'}
        # filter_dic = {'withTanh': 0, 'ActivationType': 'None'}
        # filter_dic = {'normalize_type': 'sample_wise'}
        # filter_dic = {'dataset': wm, 'aligned_prop': 0}
        # filter_dic = {'aligned_prop': 1, 'loss_sim_contras': 0.01, 'CodeTest': False}
        # filter_dic = {'dataset': '2view-caltech101-8677sample', 'aligned_prop': 0}
        # filter_dic = {'aligned_prop': 1}

        # filter_dic = {'dataset': wm, 'Epoch': 149}
        # filter_dic = {'dataset': 'NoisyMNIST30000', 'loss_sim_contras': wm}
        # filter_dic = {'seed': 1999}
        # filter_dic = {'QuickConfig': 'PengxinV2.yaml', 'seed': '9116'}

        group_by_list = []
        group_by_list = [
            'dataset', 'batch_size', 'WarmAll',
            'normalize_type', 'withTanh', 'ActivationType', 'Image01NoTanh', 'Image01AndTanh',
            'BatchNormType', 'Dropout', 'noise_type',
            'Reconstruction', 'InfoBalanceLoss', 'InfoFairLoss', 'OneHot', 'loss_sim_contras',
            'aligned_prop', 'complete_prop',
            'CodeTest', 'reFill', 'reAlign', 'reAlignK',
            # 'Desc0', 'GroupWiseLayer',
            'Rev', 'Desc0', 'GroupWiseLayer',
            'seed'
        ]
        # group_by_list = ['SeIn']
        # group_by_list = ['seed']
        # group_by_list = ['dataset']
        # group_by_list = ['batch_size', 'InfoFairLoss', 'InfoBalanceLoss', 'OneHot']
        # group_by_list = ['normalize_type', 'batch_size', 'withTanh', 'ActivationType']

        for k, v in filter_dic.items():
            df_draw = df_draw.loc[df_draw.loc[:, k] == v]
        if len(df_draw) == 0:
            warnings.warn('No one corres {}'.format(filter_dic))
            continue
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
            # dfgd = pd.DataFrame()
            # for ep in np.unique(df_draw.loc[:, 'Epoch'].values):
            #     dep = df_draw[df_draw.loc[:, 'Epoch'] == ep]
            #     dep_grp = dep.groupby(group_by_list2)
            print(group_by_list)
            dfgd = pd.concat(
                [df_draw[df_draw.loc[:, 'Epoch'] == ep].groupby(group_by_list).mean() for
                 ep in np.unique(df_draw.loc[:, 'Epoch'].values)])
            dfgd[instance_name] = dfgd.index.values
            df_draw = dfgd

        df_draw.set_index('Epoch', inplace=True)
        for name in mets_to_draw:
            if name not in df_draw.columns:
                print('{} not in df.columns'.format(name))
                continue
            xs = []
            ys = []
            for bs in np.unique(df_draw[instance_name].values):
                t = df_draw.loc[[tt == bs for tt in df_draw[instance_name].values]]
                t_d = t[name].dropna(how='any')
                if len(t_d) == 1:
                    x_draw = np.concatenate((t_d.index, t_d.index + 0.5))
                    y_draw = np.concatenate((t_d.values, t_d.values))
                elif len(t_d) > 1:
                    x_draw = t_d.index
                    y_draw = t_d.values
                else:
                    continue
                xs.append(x_draw)
                ys.append(y_draw)
            if len(xs) == 0:
                continue
            visualize_plot(
                x=xs, y=ys, labels=list(np.unique(df_draw[instance_name].values)),
                fig_path=os.path.join(root, '_fig{}{}'.format(
                    filter_dic.__str__().replace(':', ''),
                    group_by_list.__str__().replace(':', '')[:50],
                ), '{}'.format(name)),
                # xlim = [0,600]
            )
    print()


# Ac
def main():
    plot()


if __name__ == '__main__':
    main()

# (0, 'Scene15', 'None', 0.0, 0.0, 1.0, 5000.0, nan, 1.0, False, False, 100111.0, 11001.0, 'Normalize', 0.2, False, 15.0, 'None', 70.0, 996.0, nan, False, False, 0.001, 'None', 0.0, 0.0, 0.9, 0.999, 50.0, 0.0, 1.0, 999.0, 0.1, 0.04, 0.2, 0.1, 0.04, 0.2, 0.0, 'Assignment', 0.0, 'Drop', 0.1, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000000.0, False, False, 'Copy', 'Copy', 1.0, 'Unsup', nan, 0.0919983243004333, 0.0372077239980039, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, ' --ActivationType None --QuickConfig Unsup --normalize_type None --seed 996 --train_epoch 70 --withTanh 0')
# ('Scene15', 'None', 0.0, 0.0, 1.0, 5000.0, nan, 1.0, False, False, 100111.0, 11001.0, 'Normalize', 0.2, False, 15.0, 'None', 70.0, nan, False, False, 0.001, 'None', 0.0, 0.0, 0.9, 0.999, 50.0, 0.0, 1.0, 999.0, 0.1, 0.04, 0.2, 0.1, 0.04, 0.2, 0.0, 'Assignment', 0.0, 'Drop', 0.1, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000000.0, False, False, 'Copy', 'Copy', 1.0, 'Unsup', nan)
# (False, 'KnnMean', 'CorrectInfer1025')