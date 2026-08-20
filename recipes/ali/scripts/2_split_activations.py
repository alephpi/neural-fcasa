#! /usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path

import multiprocessing
from tqdm import tqdm

import numpy as np
import pandas as pd

from config import data_split

import soundfile as sf


SCRIPT_ROOT = Path(__file__).resolve().parent
CORPUS_ROOT = SCRIPT_ROOT.parent / "alicorpus"
MODE_TO_SPLIT = {
    "tr": "train",
    "cv": "eval",
    "tt": "test",
}


def resolve_data_roots(mode, rttm_root=None, audio_root=None):
    split_name = MODE_TO_SPLIT[mode]

    rttm_base = Path(rttm_root) if rttm_root else CORPUS_ROOT / "rttm"
    audio_base = Path(audio_root) if audio_root else CORPUS_ROOT / "audio" / "far"

    return rttm_base / split_name, audio_base / split_name / "audio_dir"


def split_data_one(scenario_name, duration, stepsize, dst_path, rttm_root, audio_root):
    scenario = Path(scenario_name).name
    segments = []

    speaker_to_idx = {}
    rttm_file = rttm_root / f"{scenario}.rttm"
    if not rttm_file.exists():
        raise FileNotFoundError(f"RTTM not found: {rttm_file}")

    with open(rttm_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            fields = line.split()
            if len(fields) < 8 or fields[0] != "SPEAKER":
                continue

            start = float(fields[3])
            end = start + float(fields[4])
            speaker = fields[7]

            if speaker not in speaker_to_idx:
                speaker_to_idx[speaker] = len(speaker_to_idx)
            segments.append((start, end, speaker_to_idx[speaker]))

    segments.sort(key=lambda item: item[0])

    wav_file = list(audio_root.glob(f"{scenario}*.wav"))
    if not wav_file:
        raise FileNotFoundError(f"WAV not found: {scenario}")
    wav_file = wav_file[0]
    wav_duration = sf.info(wav_file).duration

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
    scenario_path_list = list(data_split[args.mode])
    rttm_root, audio_root = resolve_data_roots(args.mode, args.rttm_root, args.audio_root)

    dst_path = CORPUS_ROOT / "processed_data" / args.mode / "act"
    dst_path.mkdir(parents=True, exist_ok=True)

    num_cores = os.cpu_count()

    # 使用 tqdm 替换 progressbar
    with multiprocessing.Pool(processes=num_cores) as pool:
        func = partial(
            split_data_one,
            duration=args.duration,
            stepsize=args.stepsize,
            dst_path=dst_path,
            rttm_root=rttm_root,
            audio_root=audio_root,
        )

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
    sub_parser.add_argument("--mode", type=str, default="tr", choices=["tr", "cv", "tt"])
    sub_parser.add_argument("--duration", type=int, default=20)
    sub_parser.add_argument("--stepsize", type=int, default=10)
    sub_parser.add_argument("--rttm-root", type=str, default=None)
    sub_parser.add_argument("--audio-root", type=str, default=None)
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
