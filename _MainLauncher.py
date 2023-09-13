import os
import time
import warnings
import datetime

import numpy as np
import pandas as pd
import torch.cuda

import _Utils.ConfigOperator as ConfigOperator
import _Utils.PathPresettingOperator as PathPresettingOperator
import _Utils.Launcher as Launcher
import _Utils.DirectoryOperator as DirectoryOperator
from _Utils.Summery import my_sota_dirs

DirectoryOperator.TestMode = False
path_operator = PathPresettingOperator.PathOperator(
    model_name='SMAIL',
    checkpoint_storage_name='SMAIL',
    code_run_set_name='SpeedF',
)
# torch.cuda.get_device_name()
cudas = ["{}".format((i + 5) % 8) for i in range(1)]
size = min(2, len(cudas) * 1)
Queue = Launcher.ProcessQueue(
    size=size,
    work_root=os.path.join(path_operator.get_code_path(level=2), '_QueueLog'),
    interval=180
)


# 1215
# 1999
# 2022
# 2023
# 996
def get_settings():
    a = []


    a += [
        [[], {'dataset': ds, 'seed': sd, 'QuickConfig': qc, 'train_epoch': 10, 'VisualFreq': 9999}]
        for sd in [1999, ]
        for ds in ['NoisyMNIST30000', 'MNISTUSPS']
        for qc in ['C100', 'A100', 'X100']
    ]


    cudalen = len(cudas)
    a = [['{}'.format(cudas[i % cudalen])] + it for i, it in enumerate(a)]
    eat = len(a) * 365.2 * 2 / 2  # NoisyMnist
    # eat = len(a) * 324  # all
    # eat = 31731.1
    print('EAT = {:.01f}s ({} proc); EndTime = {}'.format(
        eat, len(a),
        datetime.datetime.now() + datetime.timedelta(seconds=eat)))
    return a


def clear_gpu_fail(root):
    for walk_root, dirs, files in os.walk(root):
        for dr in dirs:
            txt_path = os.path.join(walk_root, dr, 'Train.txt')
            if os.path.exists(txt_path):
                txt = DirectoryOperator.FileOperator(txt_path).read()
                if 'Ending...' not in txt:
                    DirectoryOperator.FoldOperator(os.path.join(walk_root, dr)).clear()
        break


def run():
    # blocking = False
    # gun = False
    # deterministic = True
    # # if not deterministic:
    # #     warnings.warn("Not determinstic")
    # if blocking:
    #     warnings.warn("Blocking == 1 only for debug(force synchronous computation)")
    # if gun:
    #     warnings.warn("MKL_THREADING_LAYER=GNU")
    # env_config = '{}LD_LIBRARY_PATH={} {}{}'.format(
    #     'CUBLAS_WORKSPACE_CONFIG=:4096:8 ' if deterministic else '',
    #     path_operator.python_path.replace('bin/python3.9', 'lib'),
    #     'CUDA_LAUNCH_BLOCKING=1 ' if blocking else '',
    #     'MKL_THREADING_LAYER=GNU ' if gun else '',
    # )

    env_config = 'CUBLAS_WORKSPACE_CONFIG=:16:8'
    # env_config = ''
    launcher = Launcher.Launcher(path_operator=path_operator, env_config=env_config, queue=Queue)
    launcher.quick_launch(settings=get_settings(), config_operator=ConfigOperator.ConfigOperator,
                          safe_mode=False, run_file='main.py')
    # launcher.quick_launch(settings=get_settings_for_dist(), config_operator=ConfigOperator.ConfigOperator,
    #                       safe_mode=False, run_file='DistComput.py')


def main():
    # print(1908.2/4, 1177.5/2, 1098/1)
    # print(get_settings2())
    # print(path_operator.get_dataset_path(''))
    run()
    # clear_gpu_fail('/xlearning/pengxin/Codes/230516/IMvC_RunSet0517_InfoFairLoss')
    # Launcher.SystemCheckOperator().check_status()
    # Launcher.SystemCheckOperat635or().kill(2309130)
    # Launcher.SystemCheckOperator().kill(1126093)
    # Launcher.SystemCheckOperator().kill(1126095)
    # Launcher.SystemCheckOperator().kill(1126097)
    # Launcher.SystemCheckOperator().kill(1126099)
    # Launcher.SystemCheckOperator().kill(1126101)
    # Launcher.SystemCheckOperator().kill(1126103)
    # Launcher.SystemCheckOperator().kill(1126105)
    # Launcher.SystemCheckOperator().kill(1126107)
    # Launcher.SystemCheckOperator().kill(1126109)
    # Launcher.SystemCheckOperator().kill(1126111)
    # Launcher.SystemCheckOperator().kill(1126113)
    # Launcher.SystemCheckOperator().kill(1126115)
    # Launcher.SystemCheckOperator().kill(1126117)
    # Launcher.SystemCheckOperator().kill(1126119)
    # Launcher.SystemCheckOperator().kill(1126121)
    # Launcher.SystemCheckOperator().kill()
    # Launcher.SystemCheckOperator().kill()
    # Launcher.SystemCheckOperator().kill()
    # Launcher.SystemCheckOperator().__call__('rm -r -f "/xlearning/pengxin/Codes/0922/"')
    # Launcher.SystemCheckOperator().check_process_cwd(457993)
    # Launcher.SystemCheckOperator().check_process_cwd(457995)
    # Launcher.Launcher(path_operator=path_operator).tensorboard(
    #     path_to_runs='/xlearning/pengxin/Checkpoints/MoCo/0112/Ex2DxT050S6IdentInc_ImageNet100_Batch0256M3,4Worker14')


if __name__ == '__main__':
    main()
    print('Ending')

#########################################
# ToDo-使用优化：自动选择显卡 + 显式进程队列
# Todo: 加载cp之后，随机采样的随机数没有更新（应该喂50-150epoch的随机，但是喂了0-100epoch的）
