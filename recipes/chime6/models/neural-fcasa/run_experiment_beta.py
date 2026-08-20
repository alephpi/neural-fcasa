import subprocess
from itertools import zip_longest

def run_train_commands():
    # 基础命令
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_audible01_resume.yaml"
    base_command = "python train.py --multirun --config-path=./config --config-name=train_audible02_resume.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_A40.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_V100.yaml"
    
    # 定义参数值

    Gaussian_params = [ ]
    
    # leptokurtic_params = [("Leptokurtic", p) for p in leptokurtic_params]
    param_combinations = [(True, 0.5, 4.0), (True, 0.3, 4.0)]

    # 启动所有命令
    for i, (beta_prior, beta_m, beta_lmd) in enumerate(param_combinations, 1):
        # 使用f表达式构建完整命令
        full_command = f"{base_command} model.distribution=Gaussian model.dist_param=None model.beta_prior={beta_prior} model.beta_prior_m={beta_m} model.beta_prior_lmd={beta_lmd} &"
        print(f"启动命令 {i}/{len(param_combinations)}: {full_command}")
        
        # 执行命令
        subprocess.Popen(full_command, shell=True)

if __name__ == "__main__":
    run_train_commands()
