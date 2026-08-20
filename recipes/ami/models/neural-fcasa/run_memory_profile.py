import subprocess
from itertools import zip_longest

def run_train_commands():
    # 基础命令
    base_command = "python train.py --multirun --config-path=./config --config-name=train_audible01.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_audible02.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_A40.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_V100.yaml"
    

    full_command = f"{base_command} model.distribution=Gaussian model.dist_param=None trainer.max_epochs=400 &"
    print(f"启动命令 {full_command}")
    
    # 执行命令
    subprocess.Popen(full_command, shell=True)

if __name__ == "__main__":
    run_train_commands()
