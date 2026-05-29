from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Dataset, DatasetDict


DATA_ROOT = Path("/home/ids/smao-22/phd/neural-fcasa/recipes/ali/alicorpus/processed_data")
SPLITS = {
    "train": "tr",
    # "val": "cv",
    "test": "tt",
}


def build_example(wav_path, act_path, hop_length):
    wav, sr = sf.read(wav_path, dtype=np.float32)
    if wav.ndim == 1:
        wav = wav[:, None]

    duration, n_mic = wav.shape
    act = np.zeros((5, duration // hop_length), dtype=np.float32)

    df = pd.read_csv(act_path, names=("transcriber_start", "transcriber_end", "speaker_idx"))
    label_resolution = 16000 / hop_length
    for _, (start, end, spk) in df.iterrows():
        act[int(spk), int(np.floor(start * label_resolution)) : int(np.ceil(end * label_resolution))] = 1

    return {
        "wav": wav.astype(np.float32, copy=False),
        "act": act,
    }


def make_generator(split_dir, hop_length):
    derev_dir = split_dir / "derev"
    act_dir = split_dir / "act"

    wav_paths = sorted(derev_dir.glob("*.wav"))
    for wav_path in wav_paths:
        act_path = act_dir / f"{wav_path.stem}.csv"
        if not act_path.exists():
            continue
        yield build_example(wav_path, act_path, hop_length)


def main():
    parser = ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--hop-length", type=int, default=160)
    args = parser.parse_args()

    dataset_dict = {}
    for hf_split_name, split_name in SPLITS.items():
        split_dir = args.data_root / split_name
        ds = Dataset.from_generator(
            make_generator,
            gen_kwargs={"split_dir": split_dir, "hop_length": args.hop_length},
            writer_batch_size=500,
            cache_dir=str(args.data_root),
        )
        dataset_dict[hf_split_name] = ds

    dataset = DatasetDict(dataset_dict)
    dataset.save_to_disk(args.data_root / "hf_datasets")


if __name__ == "__main__":
    main()