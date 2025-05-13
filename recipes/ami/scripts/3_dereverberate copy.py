#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import partial
import os
from pathlib import Path
import concurrent.futures
import queue
import threading
from tqdm import tqdm

import cupy as cp
import librosa as lr
import soundfile as sf
from wpe import wpe


class GPUWorkerPool:
    def __init__(self, max_workers=None):
        """
        初始化 GPU 工作线程池
        
        Args:
            max_workers: 最大工作线程数，默认为 GPU 流的数量
        """
        # 获取可用 GPU 数量
        self.num_gpus = cp.cuda.runtime.getDeviceCount()
        
        # 如果未指定工作线程数，默认使用 GPU 流数量
        if max_workers is None:
            # 每个 GPU 使用多个流来并行处理
            max_workers = self.num_gpus * 4
        
        # 创建线程池
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # 创建 CUDA 流
        self.streams = [
            [cp.cuda.Stream() for _ in range(max_workers // self.num_gpus + 1)]
            for _ in range(self.num_gpus)
        ]
        
        # 输入和输出队列
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # 停止标志
        self.stop_event = threading.Event()

    def process_worker(self, worker_id):
        """
        工作线程处理函数
        
        Args:
            worker_id: 工作线程ID
        """
        # 确定 GPU 设备
        gpu_id = worker_id % self.num_gpus
        stream_id = worker_id // self.num_gpus
        
        with cp.cuda.Device(gpu_id):
            while not self.stop_event.is_set():
                try:
                    # 尝试从输入队列获取任务
                    task = self.input_queue.get(timeout=1)
                    
                    # 使用特定的 CUDA 流
                    with self.streams[gpu_id][stream_id]:
                        # 处理音频
                        result = self._process_audio(task['filename'], task['dst_path'])
                        
                        # 将结果放入输出队列
                        self.output_queue.put(result)
                        
                        # 标记任务完成
                        self.input_queue.task_done()
                
                except queue.Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    print(f"Error processing task: {e}")

    def _process_audio(self, src_filename, dst_path):
        """
        实际的音频处理逻辑
        
        Args:
            src_filename: 源音频文件路径
            dst_path: 目标输出路径
        
        Returns:
            处理结果信息
        """
        try:
            src_wav, sr = sf.read(src_filename)

            src_spec = lr.stft(src_wav.T, n_fft=512, hop_length=160)  # [M, F, T]
            src_spec = cp.asarray(src_spec)
            M, F, T = src_spec.shape

            if (cp.abs(src_spec) ** 2).max(axis=0).min() == 0:
                return None

            dst_spec = wpe(src_spec, taps=10, delay=3)

            dst_wav = lr.istft(dst_spec.get().transpose(1, 0, 2), hop_length=160).T

            sf.write(dst_path / src_filename.name, dst_wav, sr, "PCM_24")
            
            return {
                'filename': src_filename,
                'status': 'success'
            }
        
        except Exception as e:
            return {
                'filename': src_filename,
                'status': 'error',
                'error': str(e)
            }

    def process_files(self, filename_list, dst_path):
        """
        处理文件列表
        
        Args:
            filename_list: 文件名列表
            dst_path: 目标输出路径
        
        Returns:
            处理结果列表
        """
        # 启动工作线程
        workers = [
            self.executor.submit(self.process_worker, i) 
            for i in range(self.executor._max_workers)
        ]
        
        # 将文件添加到输入队列
        for filename in filename_list:
            self.input_queue.put({
                'filename': filename,
                'dst_path': dst_path
            })
        
        # 进度条
        results = []
        with tqdm(total=len(filename_list), desc="处理文件", unit="文件") as pbar:
            completed = 0
            while completed < len(filename_list):
                try:
                    # 尝试从输出队列获取结果，设置超时防止无限等待
                    result = self.output_queue.get(timeout=0.1)
                    if result:
                        results.append(result)
                    completed += 1
                    pbar.update(1)
                except queue.Empty:
                    # 队列暂时为空，继续等待
                    continue
        
        # 停止工作线程
        self.stop_event.set()
        
        # 等待所有线程结束
        concurrent.futures.wait(workers)
        
        return results

def split_data(args, unk_args):
    """
    使用多线程 GPU 并行处理音频去混响
    """
    src_filename_list = list((Path(f"./{args.mode}") / "mix").glob("*.wav"))

    dst_path = Path(f"./{args.mode}") / "derev"
    dst_path.mkdir(parents=True, exist_ok=True)

    # 创建 GPU 工作线程池并处理文件
    gpu_pool = GPUWorkerPool()
    gpu_pool.process_files(src_filename_list, dst_path)


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

    sub_parser = sub_parsers.add_parser("job", help="dereverberate mixture signals")
    sub_parser.add_argument("--mode", type=str, default="tr")
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