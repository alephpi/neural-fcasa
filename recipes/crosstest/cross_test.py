import subprocess
from pathlib import Path
from itertools import product

ROOT_DIR = Path("/home/ids/smao-22/phd/neural-fcasa")
SEP_DIAR_SCRIPT = ROOT_DIR / "neural_fcasa/separate.py"
EVAL_SCRIPT = ROOT_DIR / "recipes/crosstest/eval.py"
EVAL_DIR = ROOT_DIR / "recipes/crosstest/eval"

DATASETS = {
    "ami": {
        "src_dir": ROOT_DIR / "recipes/ami/processed_data/tt/derev",
        "ref_rttm_dir": ROOT_DIR / "recipes/ami/processed_data/tt/rttm",
        "ckpt_root": ROOT_DIR / "recipes/ami/models/neural-fcasa/outputs",
        "ckpts": {
        #    "Gaussian": "dist=Gaussian-param=None/checkpoints/last.ckpt",
        #     "Student-t_0.1": "dist=Student-t-param=0.1/checkpoints/last.ckpt",
        #     "Student-t_1.0": "dist=Student-t-param=1/checkpoints/last.ckpt",
            # "beta_prior_m=0.3-lmd=4.0":"beta_prior_m=0.3-lmd=4/checkpoints/epoch=200-val_loss=-34.0389.ckpt",
            # "beta_prior_m=0.5-lmd=4.0":"beta_prior_m=0.5-lmd=10/checkpoints/epoch=198-val_loss=-33.7312.ckpt",
            "joint": "dist=Student-t-param=1-beta_prior_m=0.5-lmd=4.0/checkpoints/last.ckpt"
        },
    },
    "ali": {
        "src_dir": ROOT_DIR / "recipes/ali/alicorpus/processed_data/tt/derev",
        "ref_rttm_dir": ROOT_DIR / "recipes/ali/alicorpus/processed_data/tt/rttm",
        "ckpt_root": ROOT_DIR / "recipes/ali/models/neural-fcasa/outputs",
        "ckpts": {
            # "Gaussian": "dist=Gaussian-param=2/checkpoints/epoch=196-val_loss=-39.5421.ckpt",
            # "Student-t_0.1": "dist=Student-t-param=0.1/checkpoints/epoch=196-val_loss=-8.3335.ckpt",
            # "Student-t_1.0": "dist=Student-t-param=1.0/checkpoints/epoch=196-val_loss=-25.0198.ckpt",
            # "beta_prior_m=0.3-lmd=4.0": "beta_prior_m=0.3-lmd=4.0/checkpoints/epoch=199-val_loss=-39.7457.ckpt",
            # "beta_prior_m=0.5-lmd=4.0": "beta_prior_m=0.5-lmd=4.0/checkpoints/epoch=199-val_loss=-39.6821.ckpt",
            "joint": "dist=Student-t-param=1-beta_prior_m=0.5-lmd=4.0/checkpoints/last.ckpt"
        },
    },
    "chime6": {
        "src_dir": ROOT_DIR / "recipes/chime6/chime6corpus/processed_data/tt/derev",
        "ref_rttm_dir": ROOT_DIR / "recipes/chime6/chime6corpus/processed_data/tt/rttm",
        "ckpt_root": ROOT_DIR / "recipes/chime6/models/neural-fcasa/outputs",
        "ckpts": {
            # "Gaussian": "dist=Gaussian-param=2/checkpoints/epoch=199-val_loss=-23.2146.ckpt",
            # "Student-t_0.1": "dist=Student-t-param=0.1/checkpoints/epoch=197-val_loss=7.7501.ckpt",
            # "Student-t_1.0": "dist=Student-t-param=1.0/checkpoints/epoch=199-val_loss=-8.7330.ckpt",
            # "beta_prior_m=0.3-lmd=4.0": "beta_prior_m=0.3-lmd=4.0/checkpoints/epoch=194-val_loss=-23.0167.ckpt",
            # "beta_prior_m=0.5-lmd=4.0": "beta_prior_m=0.5-lmd=4.0/checkpoints/epoch=186-val_loss=-23.0583.ckpt",
            "joint": "dist=Student-t-param=1-beta_prior_m=0.5-lmd=4.0/checkpoints/last.ckpt"
        }
    },
}


def sep_diar(cfg_path: Path, ckpt_path: Path, src_dir: Path, dst_dir: Path):
    if (dst_dir / ".done").exists():
        return

    dst_dir.mkdir(exist_ok=True, parents=True)

    cmd = [
        "python",
        str(SEP_DIAR_SCRIPT),
        "batch",
        str(cfg_path),
        str(ckpt_path),
        "--diarize",
        "--noi_snr=40",
        "--normalize",
        str(src_dir),
        str(dst_dir),
    ]
    print(f"exec {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    (dst_dir / ".done").touch()


def eval(diar_dir: Path, ref_rttm_dir: Path, filename: str = "score.txt"):
    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "--ref_rttm_dir",
        str(ref_rttm_dir),
        "--diar_dir",
        str(diar_dir),
    ]
    print(f"exec {' '.join(cmd)}")

    with open(diar_dir.parent / filename, "w", encoding="utf-8") as f:
        subprocess.run(cmd, check=True, stdout=f, text=True)


def resolve_ckpt_path(train_dataset: str, ckpt_path: str):
    resolved_ckpt = Path(ckpt_path)
    print(ckpt_path)
    cfg_path = resolved_ckpt.parent.parent / ".hydra" / "config.yaml"
    return cfg_path, resolved_ckpt, resolved_ckpt.parent.parent.name


def main(train_dataset, test_dataset, ckpt_path):
    # parser = argparse.ArgumentParser(
    #     description="Run separation and DER evaluation across ali/chime6 in a single script."
    # )
    # parser.add_argument("--train-dataset", choices=tuple(DATASETS.keys()), required=True)
    # parser.add_argument("--test-dataset", choices=tuple(DATASETS.keys()), required=True)
    # parser.add_argument("--ckpt-path", type=str, default=None)
    # args = parser.parse_args()

    cfg_path, ckpt_path, ckpt_label = resolve_ckpt_path(
        train_dataset,
        ckpt_path,
    )

    test_spec = DATASETS[test_dataset]
    dst_dir = (
        EVAL_DIR
        / f"{train_dataset}_to_{test_dataset}"
        / ckpt_label
        / ckpt_path.stem
        / "diar"
    )

    sep_diar(cfg_path, ckpt_path, test_spec["src_dir"], dst_dir)
    eval(dst_dir, test_spec["ref_rttm_dir"], "der.txt")


if __name__ == "__main__":
    train_test_pairs = [(train_dataset, test_dataset) for train_dataset, test_dataset in product(DATASETS.keys(), repeat=2)]
    print(train_test_pairs)
    for train_dataset, test_dataset in train_test_pairs:
        for ckpt_label, ckpt_path_suffix in DATASETS[train_dataset]["ckpts"].items():
            ckpt_path = DATASETS[train_dataset]["ckpt_root"] / ckpt_path_suffix
            main(train_dataset, test_dataset, ckpt_path)
