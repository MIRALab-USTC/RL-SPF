import argparse
import glob
import logging
import os
import queue
import re
import signal
import subprocess
import threading
import time


def detect_num_gpu():
    logger = logging.getLogger(__name__)
    device_name = re.compile(r"nvidia[0-9]")
    candidates = glob.glob("/dev/nvidia*")
    list_devices = []

    for c in candidates:
        if device_name.search(c) is not None:
            list_devices.append(c)

    logger.info("Detect {} gpus. \n{}".format(len(list_devices), list_devices))

    return len(list_devices)


class GPUTread(threading.Thread):
    def __init__(self, queue, idx, gpu_id, logdir="."):
        """

        :param queue.Queue queue:
        :param idx:
        :param gpu_id:
        :param logdir:
        """

        super().__init__(daemon=True)

        self.args_queue = queue
        self.gpu_id = gpu_id
        self.environ = os.environ.copy()
        self.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)

        self.path_stdout = os.path.join(logdir, "process_log{}.log".format(idx))
        self.path_stderr = os.path.join(logdir, "process_log{}_error.log".format(idx))

        self._exit_flag = False

    def run(self):
        logger = logging.getLogger(__name__)
        # fd_stdout = open(self.path_stdout, mode="w+")
        fd_stderr = open(self.path_stderr, mode="w+")

        logger.info("start thread on GPU{}".format(self.gpu_id))

        while True:
            if self._exit_flag or self.args_queue.empty():
                break

            args = self.args_queue.get()
            logger.info("start command {} on GPU{}".format(args, self.gpu_id))

            with subprocess.Popen(args, env=self.environ, stderr=subprocess.PIPE,
                                  preexec_fn=os.setsid, universal_newlines=True) as proc:
                self.proc = proc

                while True:
                    returncode = proc.poll()

                    # stdout = proc.stdout.readline()
                    # fd_stdout.write(stdout)
                    err = proc.stderr.readline()
                    fd_stderr.write(err)
                    fd_stderr.flush()

                    if returncode is not None:
                        break

            logger.info("finish command {} on GPU{}".format(args, self.gpu_id))

        logger.info("finish thread on GPU{}".format(self.gpu_id))

    def exit(self):
        self._exit_flag = True
        # self.proc.send_signal(signal.SIGINT)
        pgid = os.getpgid(self.proc.pid)
        # self.proc.send_signal(signal.SIGHUP)
        os.killpg(pgid, signal.SIGHUP)


def run_in_concurrent(list_args, concurrent=1, start_gpu_idx=0, available_gpus=None):
    if available_gpus is None:
        num_gpus = detect_num_gpu()
        available_gpus = list(range(num_gpus))
    num_gpus = len(available_gpus)
    assert start_gpu_idx < num_gpus

    logger = logging.getLogger(__name__ + ".{}")
    WAITTIME = 1

    environ = os.environ

    print_envs = ["CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONPATH"]

    for cur_env in print_envs:
        if cur_env in environ:
            logger.info("ENV {} : {}".format(cur_env, environ[cur_env]))

    list_threads = []

    args_queue = queue.Queue()
    for cur_args in list_args:
        args_queue.put(cur_args)

    for cur_idx in range(concurrent):
        gpu_id = available_gpus[(start_gpu_idx + cur_idx) % num_gpus]
        thread = GPUTread(queue=args_queue, idx=cur_idx, gpu_id=gpu_id)
        thread.start()
        list_threads.append(thread)
        time.sleep(WAITTIME)

    def signalHandler(signal, handler):
        logger.info("signal received")

        for thread in list_threads:
            thread.exit()

    signal.signal(signal.SIGINT, signalHandler)

    for t in list_threads:
        t.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, required=True)
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--start_gpu_idx", type=int, default=0)
    args = parser.parse_args()

    command = args.command

    if not os.path.exists(command):
        raise ValueError("{} not found".format(command))

    list_args = [command] * args.concurrent

    run_in_concurrent(list_args, concurrent=args.concurrent, start_gpu_idx=args.start_gpu_idx)


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s')
    main()
