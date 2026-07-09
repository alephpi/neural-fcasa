#! /usr/bin/env python3

from argparse import ArgumentParser
import multiprocessing
import os
from pathlib import Path
import soundfile as sf
from tqdm import tqdm

def cast_format_one(src_filename):
    wav, sr = sf.read(src_filename)
    sf.write(src_filename, wav, sr, "PCM_16")

def cast_format(args, unk_args):
    src_filename_list = list((Path(f"./processed_data/{args.mode}") / "derev").glob("*.wav"))
    num_cores = os.cpu_count()
    with multiprocessing.Pool(num_cores) as pool:
        list(tqdm(pool.imap_unordered(cast_format_one, src_filename_list), 
                  total=len(src_filename_list), 
                  desc='Processing files', 
                  unit='files')) 


def main():
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    sub_parser = sub_parsers.add_parser("job", help="cast format")
    sub_parser.add_argument("--mode", type=str, default="tt")
    sub_parser.set_defaults(handler=cast_format)
    args, unk_args = parser.parse_known_args()
    if hasattr(args, "handler"):
        args.handler(args, unk_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()