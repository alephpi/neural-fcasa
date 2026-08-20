import subprocess
from itertools import zip_longest

def run_train_commands():
    # 基础命令
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_audible01_resume.yaml"
    base_command = "python train.py --multirun --config-path=./config --config-name=train_audible02_resume.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_A40.yaml"
    # base_command = "python train.py --multirun --config-path=./config --config-name=train_V100.yaml"
    
    # 定义参数值

    # leptokurtic_params = [0.4, 0.8, 1, 1.2, 1.6]
    leptokurtic_params = []
    Gaussian_params = [2]
    student_t_params = [0.1, 1.0]
    
    # leptokurtic_params = [("Leptokurtic", p) for p in leptokurtic_params]
    Gaussian_params = [("Gaussian", p) for p in Gaussian_params]
    student_t_params = [("Student-t", p) for p in student_t_params]

    # 交替执行，保证实验多样性
    param_combinations = []
    
    for param1, param2, param3 in zip_longest(Gaussian_params, leptokurtic_params, student_t_params):
        if param1 is not None:
            param_combinations.append(param1)
        if param2 is not None:
            param_combinations.append(param2)
        if param3 is not None:
            param_combinations.append(param3)

    # 启动所有命令
    for i, (distribution, dist_param) in enumerate(param_combinations, 1):
        # 使用f表达式构建完整命令
        full_command = f"{base_command} model.distribution={distribution} model.dist_param={dist_param} model.beta_prior=False model.beta_prior_m=0.5 model.beta_prior_lmd=4.0 &"
        print(f"启动命令 {i}/{len(param_combinations)}: {full_command}")
        
        # 执行命令
        subprocess.Popen(full_command, shell=True)

if __name__ == "__main__":
    run_train_commands()
