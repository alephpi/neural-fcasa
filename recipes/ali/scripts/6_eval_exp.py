import subprocess
from pathlib import Path

ROOT_DIR= Path("/home/ids/smao-22/phd/neural-fcasa")

SRC_DIR=ROOT_DIR / "recipes/ali/alicorpus/processed_data/tt/derev/"
SEP_DIAR_SCRIPT = ROOT_DIR / "neural_fcasa/separate.py"
EVAL_SCRIPT = ROOT_DIR / "recipes/ali/scripts/5_eval.py"

def sep_diar(cfg_path: Path, ckpt_path: Path, dst_dir: Path):
    if (dst_dir/".done").exists():
        return

    dst_dir.mkdir(exist_ok=True, parents=True)

    # 构建命令列表
    cmd = [
        "python", str(SEP_DIAR_SCRIPT), "batch",
        str(cfg_path),
        str(ckpt_path),
        "--diarize",
        "--noi_snr=40",
        "--normalize",
        str(SRC_DIR),
        str(dst_dir)
    ]
    print(f"exec {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            (dst_dir/".done").touch()
    except Exception as e:
        print(e)

def eval(diar_dir: Path, filename: str = "score.txt"):
    # if (diar_dir.parent / filename).exists():
    #     return

    cmd = [
        "python",
        str(EVAL_SCRIPT),
        "--diar_dir",
        str(diar_dir)
    ]
    print(f"exec {' '.join(cmd)}")

    try:
        # 执行命令
        with open(diar_dir.parent/ filename, "w", encoding="utf-8") as f:
            subprocess.run(
                cmd,
                check=True,
                stdout=f,
                text=True
            )
    except Exception as e:
        print(e)



if __name__ == "__main__":

    CKPT_PATH_DICT = {
        # "Gaussian": "dist=Gaussian-param=2"
        # "Student-t_0.1": "dist=Student-t-param=0.1",
        "Student-t_1": "dist=Student-t-param=1.0",
    }


    dst_dir_prefix =ROOT_DIR / "recipes/ali/alicorpus/processed_data/tt/eval"
    ckpt_path_prefix = ROOT_DIR / "recipes/ali/models/neural-fcasa/outputs"
    for name, ckpt_path_suffix in CKPT_PATH_DICT.items():
        cfg_path = ckpt_path_prefix / ckpt_path_suffix / ".hydra" / "config.yaml"
        ckpt_path_dir = ckpt_path_prefix / ckpt_path_suffix / "checkpoints"
        # ckpt_paths = [p for p in ckpt_path_dir.glob("*.ckpt")]
        # ckpt_paths = [Path("/home/ids/smao-22/phd/neural-fcasa/recipes/ali/models/neural-fcasa/outputs/dist=Student-t-param=0.1/checkpoints/epoch=196-val_loss=-8.3335.ckpt")]
        ckpt_paths = [Path("/home/ids/smao-22/phd/neural-fcasa/recipes/ali/models/neural-fcasa/outputs/dist=Student-t-param=1.0/checkpoints/epoch=196-val_loss=-25.0198.ckpt")]
        for ckpt_path in ckpt_paths:
            dst_dir = dst_dir_prefix / name / ckpt_path.stem / "diar"
            sep_diar(cfg_path, ckpt_path, dst_dir)
            eval(dst_dir, "der.txt")
            # break
