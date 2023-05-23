#!/usr/bin/env python3
# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved.

# The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.

# As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.


import argparse
import logging
import os

from src.tool.task_manager import run_in_concurrent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="DDPG")
    parser.add_argument("--concurrent", type=int, default=1)
    parser.add_argument("--start_gpu_idx", type=int, default=0)
    parser.add_argument("--env", default="Walker2d-v2")
    parser.add_argument("--gin", required=True)
    parser.add_argument("--trial", type=int, default=5)
    args = parser.parse_args()

    env_name = args.env
    trial = args.trial
    path_gin = args.gin
    concurrent = args.concurrent
    start_gpu_idx = args.start_gpu_idx
    policy_name = args.policy

    if path_gin is not None:
        gin_name = os.path.basename(path_gin)
        env_base = env_name.split("-")[0]

        if not gin_name.startswith(env_base):
            raise ValueError("env doesn't match gin : {} vs {}".format(env_name, path_gin))

    default_steps = {
        "HalfCheetah-v2": 1000000,
        "Hopper-v2": 1000000,
        "Walker2d-v2": 1000000,
        "Ant-v2": 3000000,
        "Humanoid-v2": 3000000
    }
    steps = default_steps[env_name]

    args = ["python3", "src/tool/eager_main.py"]
    args += ["--env", env_name]
    args += ["--gin", path_gin]
    args += ["--steps", str(steps)]
    args += ["--policy", policy_name]

    list_args = []

    for cur_seed in range(trial):
        cur_args = args.copy()
        cur_args += ["--seed", str(cur_seed)]
        list_args.append(cur_args)

    run_in_concurrent(list_args, concurrent=concurrent, start_gpu_idx=start_gpu_idx)


if __name__ == "__main__":
    logging.basicConfig(datefmt="%d/%Y %I:%M:%S", level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s')
    main()
