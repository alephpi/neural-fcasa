import subprocess
from pathlib import Path

ROOT_DIR= Path("/home/ids/smao-22/phd/neural-fcasa")

SRC_DIR=ROOT_DIR / "recipes/ami/processed_data/tt/derev/"
SEP_DIAR_SCRIPT = ROOT_DIR / "neural_fcasa/separate.py"
EVAL_SCRIPT = ROOT_DIR / "recipes/ami/scripts/5_eval.py"

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
    # import debugpy
    # try:
    #     debugpy.listen(('localhost', 9505))
    #     print('Waiting for debugger attach')
    #     debugpy.wait_for_client()
    # except Exception as e:
    #     pass

    # CKPT_PATH_DICT = {
    #     "Gaussian": "dist=Gaussian-param=None/2025-09-02_12-50-49/0",
    #     "Laplace": "dist=Laplace-param=None/2025-09-02_12-50-49/0",
    #     "Leptokurtic_0.4": "dist=Leptokurtic-param=0.4/2025-09-03_08-31-17/0",
    #     "Leptokurtic_0.8": "dist=Leptokurtic-param=0.8/2025-09-03_08-31-18/0",
    #     "Leptokurtic_1": "dist=Leptokurtic-param=1/2025-09-03_08-31-17/0",
    #     "Leptokurtic_1.2": "dist=Leptokurtic-param=1.2/2025-09-03_08-31-17/0",
    #     "Leptokurtic_1.6": "dist=Leptokurtic-param=1.6/2025-09-03_08-31-17/0",
    #     "Student-t_0.1": "dist=Student-t-param=0.1/2025-09-03_08-31-17/0",
    #     "Student-t_1": "dist=Student-t-param=1/2025-09-03_08-31-18/0",
    #     "Student-t_10": "dist=Student-t-param=10/2025-09-03_08-31-18/0",
    #     "Student-t_100": "dist=Student-t-param=100/2025-09-03_08-31-17/0"
    # }

    CKPT_PATH_DICT = {
        "Leptokurtic_0.4": "dist=Leptokurtic-param=0.4/2025-09-11_18-51-11/0",
        "Leptokurtic_0.8": "dist=Leptokurtic-param=0.8/2025-09-11_18-51-11/0",
        "Leptokurtic_1": "dist=Leptokurtic-param=1/2025-09-11_18-51-10/0",
        "Leptokurtic_1.2": "dist=Leptokurtic-param=1.2/2025-09-11_18-51-10/0",
        "Leptokurtic_1.6": "dist=Leptokurtic-param=1.6/2025-09-11_18-51-10/0",
        "Student-t_0.1": "dist=Student-t-param=0.1/2025-09-11_18-51-11/0",
        "Student-t_1": "dist=Student-t-param=1/2025-09-11_18-51-11/0",
        "Student-t_10": "dist=Student-t-param=10/2025-09-11_18-51-10/0",
        "Student-t_100": "dist=Student-t-param=100/2025-09-11_18-51-11/0"
    }

    dst_dir_prefix =ROOT_DIR / "recipes/ami/processed_data/tt/new_eval"
    ckpt_path_prefix = ROOT_DIR / "recipes/ami/models/neural-fcasa/outputs"
    for name, ckpt_path_suffix in CKPT_PATH_DICT.items():
        cfg_path = ckpt_path_prefix / ckpt_path_suffix / ".hydra" / "config.yaml"
        ckpt_path_dir = ckpt_path_prefix / ckpt_path_suffix / "checkpoints"
        ckpt_paths = [p for p in ckpt_path_dir.glob("*.ckpt")]
        for ckpt_path in ckpt_paths:
            dst_dir = dst_dir_prefix / name / ckpt_path.stem / "diar"
            sep_diar(cfg_path, ckpt_path, dst_dir)
            eval(dst_dir, "der.txt")
            # break
