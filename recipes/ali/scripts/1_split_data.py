#!/usr/bin/env python3
from argparse import ArgumentParser
from functools import partial
import multiprocessing
import os
from pathlib import Path
from tqdm import tqdm  # 替换 progressbar
import numpy as np
from config import data_split, dataset_path
import soundfile as sf

def split_data_one(scenario_file, duration, stepsize, dst_path):
    scenario_basename = scenario_file.stem.rsplit("_",1)[0]
    wav_filename = scenario_file

    wav, sr = sf.read(wav_filename)

    for tidx, t_start in enumerate(range(0, wav.shape[0], sr * stepsize)):
        if wav.shape[0] < (t_end := t_start + sr * duration):
            break

        sf.write(dst_path / f"{scenario_basename}.{tidx:03d}.wav", wav[t_start:t_end], sr)


def split_data(args, unk_args):
    scenario_file_list = []
    for scenario_basename in data_split[args.mode]:
        if args.mode == "tr":
            dataset_path_ = dataset_path / "train" / "audio_dir"
        elif args.mode == "cv":
            dataset_path_ = dataset_path / "eval" / "audio_dir"
        elif args.mode == "tt":
            dataset_path_ = dataset_path / "test" / "audio_dir"
        else:
            raise ValueError(f"unknown mode: {args.mode}")
        scenario_file_list += list(dataset_path_.glob(f"{scenario_basename}*"))
    # print(scenario_file_list)

    dst_path = Path(f"./alicorpus/processed_data/{args.mode}") / "mix"
    dst_path.mkdir(parents=True, exist_ok=True)

    num_cores = os.cpu_count()

    # 使用 tqdm 替换 progressbar
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(split_data_one, duration=args.duration, stepsize=args.stepsize, dst_path=dst_path)

        # 使用 tqdm 包装 pool.imap
        list(tqdm(pool.imap(func, scenario_file_list), 
                  total=len(scenario_file_list), 
                  desc='Processing scenarios', 
                  unit='scenario'))

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
            f.write("#SBATCH --partition=CPU\n")  # 使用40个CPU核心
            f.write("#SBATCH --time=3:00:00\n")
            # 直接运行Python脚本
            f.write(f"srun python ./scripts/{command_name}.py job --mode {mode} ")
            f.write(" ".join(unk_args) + "\n")

        os.system(f"sbatch {filename_job}")

def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("job", help="split data")
    sub_parser.add_argument("--mode", type=str, default="tr")
    sub_parser.add_argument("--duration", type=int, default=20)
    sub_parser.add_argument("--stepsize", type=int, default=10)
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
