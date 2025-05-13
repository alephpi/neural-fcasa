#! /usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path
from xml.etree import ElementTree

import multiprocessing
from tqdm import tqdm

import numpy as np
import pandas as pd

from config import data_split, dataset_path, metadata_path

import soundfile as sf


def split_data_one(scenario_path, duration, stepsize, dst_path):
    scenario = scenario_path.name
    if scenario in ["IS1003b", "IS1007d"]:
        return

    segments = []
    for xml_filename in (metadata_path / "segments").glob(f"{scenario}*.xml"):
        tree = ElementTree.parse(xml_filename)
        for child in tree.getroot():
            start = float(child.attrib["transcriber_start"])
            end = float(child.attrib["transcriber_end"])

            segments.append((start, end, int(child.attrib["channel"])))
    segments.sort(key=lambda item: item[0])

    wav_duration = sf.info(scenario_path / "audio" / f"{scenario}.Array1-01.wav").duration
    for tidx, t_start in enumerate(np.arange(0, wav_duration, stepsize)):
        if wav_duration < (t_end := t_start + duration):
            break

        target_segments = []
        for seg in segments:
            if seg[0] <= t_start and t_end <= seg[1] or t_start <= seg[0] < t_end or t_start < seg[1] <= t_end:
                target_segments.append((max(seg[0] - t_start, 0), min(seg[1] - t_start, t_end - t_start), seg[2]))

        df = pd.DataFrame(target_segments, columns=("transcriber_start", "transcriber_end", "speaker_idx"))
        df.to_csv(dst_path / f"{scenario}.{tidx:03d}.csv", header=False, index=False)


def split_data(args, unk_args):
    scenario_path_list = []
    for scenario_basename in data_split[args.mode]:
        scenario_path_list += list(dataset_path.glob(f"{scenario_basename}*"))

    dst_path = Path(f"./processed_data/{args.mode}") / "act"
    dst_path.mkdir(parents=True, exist_ok=True)

    num_cores = os.cpu_count()

    # 使用 tqdm 替换 progressbar
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(split_data_one, duration=args.duration, stepsize=args.stepsize, dst_path=dst_path)

        # 使用 tqdm 包装 pool.imap
        list(tqdm(pool.imap(func, scenario_path_list), 
                  total=len(scenario_path_list), 
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

    with open(f"{dataset_path}/scripts/job_template.sh") as f:
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
