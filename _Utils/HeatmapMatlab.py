import numpy as np
import pandas as pd


def csv_to_heat_maps(df, x_name, y_name, z_names):
    x = np.unique(df.loc[:, x_name])
    y = np.unique(df.loc[:, y_name])
    for it in [x, y]:
        for v in it:
            print(v, end=',')
        print()

    for z_name in z_names:
        z = np.zeros((x.__len__(), y.__len__()))
        for i, x_val in enumerate(x):
            for j, y_val in enumerate(y):
                ind = np.logical_and(df.loc[:, x_name] == x_val, df.loc[:, y_name] == y_val)
                assert np.sum(ind) == 1
                z[i, j] = np.round(df.loc[ind, z_name], 2)

        # plot_heat_map(z=z, ylabel='Fair', xlabel='ClusterFavorable', title='ACC', xticks=u_clu, yticks=u_fair,
        #               show=True, fig_path=pt)

        # z = z[:, ::-1].transpose()

        for i, f in enumerate(x):
            for j, c in enumerate(y):
                print(z[i, j], end='')
                if j == len(z[0]) - 1:
                    print('', end=';')
                else:
                    print('', end=',')
            print()
        print()


def main():
    src = r'D:\Pengxin\Workplace\Lab\Multi-Clustering\ExpTemp.csv'
    df = pd.read_csv(src)  # df = pre_process_df(

    csv_to_heat_maps(df, x_name='InfoBalanceLoss', y_name='loss_sim_contras',
                     z_names=['Multi-acc', 'Multi-nmi', 'Multi-ari', ])


if __name__ == '__main__':
    main()
