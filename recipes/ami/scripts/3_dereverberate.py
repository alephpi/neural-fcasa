#! /usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path

import multiprocessing
import sys
from tqdm import tqdm

import cupy as cp
from wpe import wpe

import librosa as lr
import soundfile as sf

import debugpy
try:
    debugpy.listen(("localhost", 9500))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    pass

def split_data_one(src_filename, dst_path):
    """
    Note that we used our torch implementation of WPE in our Interspeech paper,
    while here we replaced it with the more standard `gpu-wpe`.
    If there is a reproduction issue, please let us know.
    """
    rank = 0

    with cp.cuda.Device(rank):
        src_wav, sr = sf.read(src_filename)

        src_spec = lr.stft(src_wav.T, n_fft=512, hop_length=160)  # [M, F, T]
        src_spec = cp.asarray(src_spec)
        M, F, T = src_spec.shape

        if (cp.abs(src_spec) ** 2).max(axis=0).min() == 0:
            # if it is silent somewhere, we just output the original audio
            # dst_wav = src_wav
            # sf.write(dst_path / src_filename.name, dst_wav, sr, "PCM_24")
            return

        dst_spec = wpe(src_spec.transpose(1,0,2), taps=10, delay=3)

        dst_wav = lr.istft(dst_spec.get().transpose(1, 0, 2), hop_length=160).T
    sf.write(dst_path / src_filename.name, dst_wav, sr, "PCM_16")

def safe_split_data_one(src_filename, dst_path):
    try:
        return split_data_one(src_filename, dst_path)
    except Exception as e:
        print(f"Error processing {src_filename}: {e}")

def split_data(args, unk_args):
    src_filename_list_all = list((Path(f"./processed_data/{args.mode}") / "mix").glob("*.wav"))
    exclude_list = list((Path(f"./processed_data/{args.mode}") / "derev").glob("*.wav"))
    exclude_list = [p.name for p in exclude_list]
    src_filename_list = [p for p in src_filename_list_all if not p.name in exclude_list]


    dst_path = Path(f"./processed_data/{args.mode}") / "derev"
    dst_path.mkdir(parents=True, exist_ok=True)

    num_cores = 8

    # 使用 tqdm 替换 progressbar
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(split_data_one, dst_path=dst_path)

        # 使用 tqdm 包装 pool.imap
        list(tqdm(pool.imap_unordered(func, src_filename_list), 
                  total=len(src_filename_list), 
                  desc='Processing files', 
                  unit='files'))


def submit_jobs(args, unk_args):
    script_path = Path(__file__)
    dataset_path = script_path.parent.parent
    command_name = script_path.stem

    job_path = Path(f"jobs/{command_name}/")
    out_path = Path(f"jobs.out/{command_name}/")
    job_path.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(f"{dataset_path}/scripts/slurm_job_template.sh") as f:
        job_template = f.read()

    for mode in ["tr", "cv", "tt"]:
        filename_job = job_path / f"{mode}.sh"
        filename_stdout = out_path / f"{mode}.out"
        filename_stderr = out_path / f"{mode}.err"

        with open(filename_job, "w") as f:
            f.write(job_template)
            f.write(f"#SBATCH --output={filename_stdout}\n")
            f.write(f"#SBATCH --error={filename_stderr}\n")
            f.write("#SBATCH --nodes=1\n")  # 单节点
            f.write("#SBATCH --ntasks=1\n")  # 单任务
            f.write("#SBATCH --cpus-per-task=40\n")  # 使用40个CPU核心
            f.write("#SBATCH --partition=V100-32G\n")  # 使用40个CPU核心
            f.write("#SBATCH --time=3:00:00\n")
            f.write(f"")
            # 直接运行Python脚本
            f.write(f"srun python ./scripts/{command_name}.py job --mode {mode} ")
            f.write(" ".join(unk_args) + "\n")

        # os.system(f"sbatch {filename_job}")

def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("job", help="dereverberate mixture signals")
    sub_parser.add_argument("--mode", type=str, default="tr")
    sub_parser.set_defaults(handler=split_data)

    sub_parser = sub_parsers.add_parser("sub", help="submit jobs")
    sub_parser.set_defaults(handler=submit_jobs)

    args, unk_args = parser.parse_known_args()
    if hasattr(args, "handler"):
        args.handler(args, unk_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
