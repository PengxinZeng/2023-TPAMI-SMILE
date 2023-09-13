import time

from _MainLauncher import path_operator
from _Utils import Launcher
from _Utils.ConfigOperator import ConfigOperator


def main():
    class C2(ConfigOperator):
        def get_name(self, *args, **kwargs):
            return '_QueueLog'

    Launcher.Launcher(
        path_operator=path_operator,
        env_config=''
    ).launch(
        run_file='_MainLauncher.py',
        cfg=C2(cuda='0'),
        model_name='Train{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))),
        safe_mode=False,
        clean_fold=False,
    )


if __name__ == '__main__':
    main()
