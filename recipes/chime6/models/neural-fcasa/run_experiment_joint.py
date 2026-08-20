import subprocess
from itertools import zip_longest

def run_train_commands():
    # 基础命令
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_audible01_resume.yaml"
    base_command = "python train.py --multirun --config-path=./config --config-name=train_audible02_resume.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_A40.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_V100.yaml"
    
    # 定义参数值

    # 启动所有命令
    # 使用f表达式构建完整命令
    full_command = f"{base_command} model.distribution=Student-t model.dist_param=1 model.beta_prior=True model.beta_prior_m=0.5 model.beta_prior_lmd=4.0 &"
    print(f"启动命令 {full_command}")
        
        # 执行命令
    subprocess.Popen(full_command, shell=True)

if __name__ == "__main__":
    run_train_commands()
