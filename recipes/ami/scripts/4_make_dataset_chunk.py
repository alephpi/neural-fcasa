#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
from math import ceil, floor
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from multiprocessing import Pool
from tqdm import tqdm
import h5py

class EmptySample(Exception):
    pass

def process_one_file(wav_fname, mode_path, args):
    try:
        if wav_fname is None:
            raise EmptySample()

        # 加载音频文件
        wav, sr = sf.read(wav_fname, dtype=np.float32)
        duration, n_mic = wav.shape

        # 加载对应的CSV文件
        csv_fname = mode_path / "act" / f"{wav_fname.stem}.csv"
        if not csv_fname.exists():
            raise EmptySample()

        df = pd.read_csv(csv_fname, names=("transcriber_start", "transcriber_end", "speaker_idx"))

        # 生成活动矩阵
        # 默认label resolution 为 10ms
        label_resolution = 16000 / args.hop_length
        # 至多有5名说话人
        act = np.zeros([5, duration // args.hop_length], dtype=np.float32)
        for _, (start, end, spk) in df.iterrows():
            act[int(spk), floor(start * label_resolution) : ceil(end * label_resolution)] = 1

        return wav_fname.stem, wav, act, duration, n_mic
    except Exception as e:
        print(f"Error processing {wav_fname}: {e}")
        return None

def make_dataset(args, unk_args):
    mode_path = Path(f"./processed_data/{args.mode}/")

    print("================================")
    print("Parameters")
    print("--------------------------------")
    for key, val in args.__dict__.items():
        print(f"{key:20s}: {val}")
    print("================================")

    # 获取文件列表
    print("init")
    wav_fname_list = sorted((mode_path / args.data).glob("*.wav"))

    os.makedirs("./processed_data/hdf5", exist_ok=True)
    hdf_name = f"./processed_data/hdf5/chunk.{args.data}-hop{args.hop_length}-{args.mode}.hdf5"
    n_cpus = 8
    func = partial(process_one_file, mode_path=mode_path, args=args)

    with h5py.File(hdf_name, "w") as f:
        with Pool(processes=n_cpus) as pool:
            for result in tqdm(pool.imap(func, wav_fname_list), total=len(wav_fname_list), desc="Processing files"):
                if result is None:
                    continue
                grp_name, wav, act, duration, n_mic = result
                g = f.create_group(grp_name)
                g.create_dataset("wav", [n_mic, duration], "float32")
                g.create_dataset("act", [5, duration // args.hop_length], "float32")

                f[f"{grp_name}/wav"][:] = wav.T
                f[f"{grp_name}/act"][:] = act


def submit_jobs(args, unk_args):
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    command_name = script_path.stem

    job_path = Path(f"jobs/{command_name}/")
    out_path = Path(f"jobs.out/{command_name}/")
    job_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(f"{dataset_path}/scripts/job_template.sh") as f:
        job_template = f.read()

    for mode in ["tr", "cv"]:
        fname_job = f"jobs/{command_name}/{mode}.sh"
        fname_stdout = f"jobs.out/{command_name}/{mode}.out"

        with open(fname_job, "w") as f:
            f.write(job_template)
            f.write(f"python ./scripts/{command_name}.py gen --mode {mode} ")
            f.write(" ".join(unk_args) + "\n")

        os.system(f"qsub -g $JOB_GROUP $QSUB_ARGS -l rt_F=4 -l h_rt=1:0:0 -o {fname_stdout} {fname_job}")

def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("gen", help="generate hdf5")
    sub_parser.add_argument("--mode", type=str, default="tr")
    sub_parser.add_argument("--data", type=str, default="derev")
    sub_parser.add_argument("--hop_length", type=int, default=160)
    sub_parser.set_defaults(handler=make_dataset)

    sub_parser = sub_parsers.add_parser("sub", help="submit jobs")
    sub_parser.set_defaults(handler=submit_jobs)

    args, unk_args = parser.parse_known_args()
    if hasattr(args, "handler"):
        args.handler(args, unk_args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()