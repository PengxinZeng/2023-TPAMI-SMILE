import os

import numpy as np
import pandas as pd
from _Utils.DirectoryOperator import DirectoryOperator


def merge_csv(dt_old, dt):
    for ind in dt.index:
        for na, v in dt.loc[ind].items():
            if isinstance(v, list):
                v = '{}'.format(v)
            # if isinstance(v, np.ndarray) and len(v) > 10:
            #     continue
            dt_old.loc[ind, na] = v
    return dt_old


def update_log(dic, path='../log/res.csv'):
    index = 'Epoch'
    val = []
    name = []
    for na, v in dic.items():
        val.append(v)
        name.append(na)
    dt = pd.DataFrame([val], columns=name)
    dt = dt.set_index(index)
    if os.path.exists(path):
        dt_old = pd.read_csv(path, index_col=index)
        dt = merge_csv(dt_old, dt)
    DirectoryOperator(path).make_fold()
    dt.to_csv(path)


if __name__ == '__main__':
    update_log({'acc': 1, 'epoch': 2})
    update_log({'acc': 2, 'epoch': 3})
    update_log({'epoch': 3, 'nmi': 4})
    update_log({'epoch': 3, 'nmi': 5})
    update_log({'epoch': 4, 'nmi': 5, 'acc': 6})
    update_log({'epoch': 2, 'nmi': 5, 'acc': 7})
