import subprocess
import torch
import threading
import time
import psutil

def get_gpu_utilization():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE
    )
    try:
        utilization = result.stdout.decode('utf-8').strip().split('\n')
        utilization = [float(u) for u in utilization if u]
        return sum(utilization) / len(utilization) if utilization else 0.0
    except ValueError:
        return 0.0

def monitor_resources(stop_event, cpu_percentages, memory_percentages, gpu_percentages):
    while not stop_event.is_set():
        cpu_percentages.append(psutil.cpu_percent(interval=0.1))
        memory_percentages.append(psutil.virtual_memory().percent)
        if torch.cuda.is_available():
            gpu_percentages.append(get_gpu_utilization())
        time.sleep(0.1) 

def evaluate_model(trainer):
    cpu_percentages = []
    memory_percentages = []
    gpu_percentages = []

    stop_event = threading.Event()

    monitor_thread = threading.Thread(target=monitor_resources, args=(stop_event, cpu_percentages, memory_percentages, gpu_percentages))
    monitor_thread.start()

    eval_start_time = time.time()
    eval_result = trainer.evaluate()
    eval_end_time = time.time()

    stop_event.set()
    monitor_thread.join()

    avg_cpu = sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0.0
    avg_memory = sum(memory_percentages) / len(memory_percentages) if memory_percentages else 0.0
    avg_gpu = sum(gpu_percentages) / len(gpu_percentages) if gpu_percentages else 0.0


    total_time = eval_end_time - eval_start_time

    return {
        'eval_result': eval_result,
        'avg_cpu_utilization': avg_cpu,
        'avg_memory_utilization': avg_memory,
        'avg_gpu_utilization': avg_gpu,
        'total_time': total_time
    }
