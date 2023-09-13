import json
import os
import subprocess
import time
import warnings
from shutil import ignore_patterns


import numpy as np
import sys
import _Utils.DirectoryOperator as DirectoryOperator
from _Utils.Mail import send
from _Utils.TimeOperator import Timer


class SubprocessOperator:
    def __call__(self, cmd):
        print(cmd)
        if not DirectoryOperator.TestMode:
            res = subprocess.run(cmd, shell=True)
            # print(res)


class SystemCheckOperator(SubprocessOperator):
    def __init__(self):
        super(SystemCheckOperator, self).__init__()

    def check_status(self):
        print()
        if sys.platform == 'win32':
            self.__call__(cmd='tasklist')
        else:
            self.__call__(cmd='ps -ef|grep python')
            print()
            self.__call__(cmd='ps -ef|grep pengxin')
            print()
            self.__call__(cmd='top -n 1')
        self.__call__(cmd='nvidia-smi')

    def check_process_cwd(self, pid=None):
        if pid is not None:
            self.__call__(cmd='cd /proc/{} &&  ls -l cwd'.format(pid))

    def check_processor(self):
        self.__call__(cmd='cat /proc/cpuinfo |grep "processor"')

    def check_disk(self):
        self.__call__(cmd='lsblk -d -o name,rota')
        print()
        self.__call__(cmd='df -h')
        print()
        self.__call__(cmd='cd {} && du -h --max-depth=1'.format('./'))

    def kill(self, pid=None):
        if pid is not None:
            if sys.platform == 'win32':
                self.__call__(cmd='Taskkill /PID {} /F'.format(pid))
            else:
                self.__call__(cmd='kill {}'.format(pid))


class Launcher(SubprocessOperator):
    def __init__(self, path_operator, env_config='', queue=None):
        self.path_operator = path_operator
        self.env_config = env_config
        self.queue = queue

    def show_tensorboard(self, path_to_runs):
        python_path = self.path_operator.python_path
        tensorboard_path = os.path.join(os.path.dirname(python_path), 'tensorboard')
        # self.__call__(cmd='find \'{}\' | grep tfevents'.format(path_to_runs))
        # self.__call__(cmd='{} {} --inspect --logdir \'{}\''.format(python_path, tensorboard_path, path_to_runs))
        self.__call__(cmd='{} {} --logdir \'{}\' {}'.format(
            python_path, tensorboard_path, path_to_runs, self.path_operator.tensorboard_arg))

    def launch(self, cfg, run_file="main.py", safe_mode=True, model_name='Train', clean_fold=True):
        fold_path = self.path_operator.get_code_path(code_fold_name=cfg.get_name(), level=3)
        if os.path.exists(os.path.join(fold_path, 'Checkpoints')):
            warnings.warn('There are some checkpoints in "{}".'.format(fold_path))
        if clean_fold:
            DirectoryOperator.FoldOperator(directory=fold_path).clear(delete_root=False, not_to_delete_file=safe_mode)
        code_root = os.path.join(fold_path, '{}Code'.format(model_name))
        # if sys.platform != 'win32':
        #     code_root = '"{}"'.format(code_root)
        DirectoryOperator.FoldOperator(directory='./').copy(
            dst_fold=code_root, not_to_delete_file=safe_mode, ignore=ignore_patterns('__pycache__', '.idea', '_bac'))
        python_cmd = '{} -u {} {}'.format(
            self.path_operator.python_path,
            # np.random.randint(0,1000),
            run_file,
            cfg.get_config(),
        )
        txt_path = '"{:}.txt"'.format(os.path.join(fold_path, model_name))
        if sys.platform != 'win32':
            py_cmd = "{append_config:} nohup {python_cmd:} > {txt_path:} 2>&1 &".format(
                append_config=self.env_config + ' CUDA_VISIBLE_DEVICES={}'.format(cfg.cuda),
                python_cmd=python_cmd,
                txt_path=txt_path,
            )
        else:
            py_cmd = "start /b {python_cmd:} > {txt_path:} 2>&1".format(
                python_cmd=python_cmd,
                txt_path=txt_path,
            )
        self.__call__(
            cmd="cd \"{code_root:}\" && {py_cmd:}".format(
                code_root=code_root,
                py_cmd=py_cmd,
            )
        )

    def quick_launch(self, settings, config_operator, **kwargs):
        """
        settings: [cuda, [yaml_list], arg_dict]
        """
        timer = Timer()
        total_count = len(settings)
        for cuda, yaml_list, arg_dict in settings:
            t_start = time.time()
            cfg = config_operator(
                yaml_list=yaml_list,
                cuda=cuda,
                **arg_dict,
            )
            work_root = self.path_operator.get_code_path(code_fold_name=cfg.get_name(), level=3)
            if os.path.exists(work_root):
                print("Skipping a code running due to its log already exists. work_root == {}".format(work_root))
                total_count -= 1
                continue
            if self.queue is not None:
                self.queue.enqueue(work_root=work_root, cuda=cuda)
            self.launch(
                cfg=cfg,
                **kwargs
            )
            timer.update(time.time()-t_start)
            timer.count -= 1
            timer.show(total_count=total_count)
            timer.count += 1
        if self.queue is not None:
            self.queue.close()
class Cuda:
    def __init__(self, work_root):
        self.subprocess_root = os.path.join(work_root, 'Subprocess')

class Process:
    def __init__(self, start_time=None, work_root=None, cuda=None, pcb_file=None):
        self.start_time = start_time
        self.work_root = work_root
        self.cuda = cuda
        self.pcb_file = pcb_file

    def from_json(self, src):
        with open(src, 'r') as f:
            self.__dict__ = json.load(f)
        return self

    def to_json(self):
        DirectoryOperator.FileOperator(self.pcb_file).make_fold()
        if not DirectoryOperator.TestMode:
            with open(self.pcb_file, 'w') as f:
                json.dump(self.__dict__, f)
                # print('created {}'.format(self.pcb_file))


class ProcessQueue:
    def __init__(self, size, work_root, interval=180):
        self.size = size
        self.work_root = work_root
        self.subprocess_root = os.path.join(self.work_root, 'Subprocess')
        self.count = 0
        self.start_time = time.time()
        self.interval = interval

    def enqueue(self, work_root=os.path.abspath('../'), cuda=None):
        # if os.path.exists(self.subprocess_root):
        #     dir_list = os.listdir(self.subprocess_root)
        #     cnt = len(dir_list)
        #     if self.room_size == self.size and cnt > 0:
        #         time.sleep(120)
        while True:
            if os.path.exists(self.subprocess_root):
                dir_list = os.listdir(self.subprocess_root)
                if 'END' in dir_list:
                    os.remove(os.path.join(self.subprocess_root, 'END'))
                    print("'END' in dir_list, removed it and exit.")
                    exit(1)
                    # raise AssertionError("'END.txt' in dir_list, removed it and exit.")
                time_now = time.time()
                for f in dir_list:
                    mt = DirectoryOperator.DirectoryOperator('{}'.format(
                        os.path.join(Process().from_json(os.path.join(self.subprocess_root, f)).work_root,
                                     'Train.txt'))

                    ).modification_time()
                    t = time_now - mt
                    # print('time_now == {}'.format(time_now))
                    # print('mt == {}'.format(mt))
                    # print('t == {}'.format(t))
                    # print('self.interval == {}'.format(self.interval))
                    if t > self.interval:
                        send(
                            mail_title="Pengxin's Mail Notification",
                            mail_text="One of your code has been running for {interval:.02f}s without any progress, and been treat to be died.\n\t--From {sur}".format(
                                interval=t,
                                sur=self.subprocess_root
                            )
                        )
                        print('removing {}'.format(os.path.join(self.subprocess_root, f)))
                        DirectoryOperator.FileOperator(os.path.join(self.subprocess_root, f)).remove()
                dir_list = os.listdir(self.subprocess_root)
                cnt = len(dir_list)
            else:
                cnt = 0
            # print(cnt)
            # print(os.listdir(self.subprocess_root))
            if cnt < self.size:
                start_time = time.time()
                time_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(start_time))
                fn = os.path.join(
                    self.subprocess_root,
                    'Process{}_{:03d}{:03d}.json'.format(time_str, self.count, cnt)
                )

                print('{time} Enqueue {fn}'.format(time=time_str,
                                                   fn=fn))
                # DirectoryOperator.FileOperator(fn).write(data='...')

                Process(start_time=start_time, work_root=work_root, cuda=cuda, pcb_file=fn).to_json()
                self.count += 1
                break
            else:
                time.sleep(10)

    def dequeue(self):
        work_root = os.path.abspath('../')
        if os.path.exists(self.subprocess_root):
            for fn in np.sort(os.listdir(self.subprocess_root)):
                wr = Process().from_json(os.path.join(self.subprocess_root, fn)).work_root
                if wr == work_root:
                    print('{time} Dequeue {fn}'.format(
                        time=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                        fn=fn))
                    DirectoryOperator.FileOperator(os.path.join(self.subprocess_root, fn)).remove()
                    return
        warnings.warn('Wrong dequeue path; {}'.format(work_root))
        send(
            mail_title="Pengxin's Mail Notification",
            mail_text="Your code encountered wrong dequeue path or finished without queue.\n\t--From {}".format(
                self.subprocess_root
            )
        )

    def close(self):
        self.size = 1
        self.enqueue()
        self.dequeue()
        total_time = time.time() - self.start_time
        pro_count = self.count - 1
        send(
            mail_title="Pengxin's Mail Notification",
            mail_text="Your code finished in {:.02f}s ({}pro * {:.02f}s/pro).\n\t--From {}".format(
                total_time,
                pro_count,
                total_time / pro_count,
                self.subprocess_root
            )
        )
