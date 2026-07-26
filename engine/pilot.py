# Lets program run commands or other programs as though you typed them into terminal
# By letting code run another program, we can capture info about what it does
import subprocess

# Gives access to info and controls related to interpreter and program currently running
import sys

# JSON is common text format used to store and send structured data
import json

# Math module has many built-in mathematical functions
import math

# Used to get information about your computer’s running processes and system resources
import psutil

# Loads Python’s interface to NVIDIA Management Library (NVML)
# Lets program ask NVIDIA GPU for information such as: GPU utilization, GPU memory usage, power usage, etc
import pynvml

# Attempts to start and initialize NVIDIA’s GPU-monitoring library so your program 
# can communicate with the NVIDIA driver and inspect the GPUs
try:
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

def parse_pilot_output(raw_output):
    """Turn raw stdout text (1 JSON line per step) into a list of real Python
    dictionaries"""

    steps = []

    for line in raw_output.splitlines():

        # Skip blank lines
        if line.strip():

            # Takes the string and converts it into normal Python data
            data = json.loads(line)

            # Extract dataset info
            if data.get("type") == "dataset_info":
                dataset_info = data
            else:
                steps.append(data)

    return dataset_info, steps

def compute_average_timings(steps, warmup_steps=1):
    """Skip the first `warmup_steps` steps (startup noise), then compute
    the average of each timing field across the remaining steps"""

    # Throw away the first/first few steps before computing average
    # Running the first 10 ms for example doesn't give us valuable info
    measured_steps = steps[warmup_steps:]

    if not measured_steps:
        return {}

    # Goal is to determine which phase takes the most time on average
    timing_fields = ["data_loading_ms", "transfer_ms", "forward_ms", "backward_ms", "optimizer_ms"]

    averages = {}

    # Iterate through each dictionary (step)
    for field in timing_fields:
        values = [step[field] for step in measured_steps]
        averages[field] = sum(values) / len(values)
    
    return averages

def compute_total_step_time(averages):
    """Sum all phase averages into one total average time per step (ms)"""

    return sum(averages.values())

def estimate_full_run_time(averages, total_steps):
    """Estimate total training time (in minutes) given the average
    per-step time and how many steps the full run will take"""

    total_step_time_ms = compute_total_step_time(averages)
    total_time_ms = total_step_time_ms * total_steps
    total_time_minutes = total_time_ms / 1000 / 60

    return total_time_minutes

def compute_steps_per_epoch(dataset_info):
    """Given dataset_info (with dataset_size and batch_size), compute
    how many steps (batches) make up one full epoch)"""

    if dataset_info is None:
        return None
    
    dataset_size = dataset_info["dataset_size"]
    batch_size = dataset_info["batch_size"]

    return math.ceil(dataset_size / batch_size)

def diagnose_bottleneck(averages, threshold=0.4):
    """Given average per-phase timings identify which phase (if any)
    takes up more than `threshold` fraction of total step time"""

    total_time = compute_total_step_time(averages)

    if total_time == 0:
        return None

    bottleneck = None
    highest_fraction = 0

    for phase, avg_time in averages.items():
        fraction = avg_time / total_time

        # Find largest fraction
        if fraction > highest_fraction:
            highest_fraction = fraction
            bottleneck = phase

    # Needs to be larger than threshold
    if highest_fraction >= threshold:
        return {"phase": bottleneck, "fraction": highest_fraction}
    else:
        return None

def run_pilot_with_monitoring(filepath, steps=20, poll_interval=0.2):
    """Launch a training script as a subprocess, and periodically sample
    its CPU and memory usage while it runs"""

    process = subprocess.Popen(
        [sys.executable, filepath, "--steps", str(steps)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # psutil.Process lets us query stats about a specific running process,
    # given its process ID (process.pid, provided automatically by Popen)
    ps_process = psutil.Process(process.pid)

    cpu_samples = []
    memory_samples = []

    # None means "still running"
    while process.poll() is None:  
        try:
            cpu_percent = ps_process.cpu_percent(interval=poll_interval)
            memory_mb = ps_process.memory_info().rss / (1024 * 1024)

            cpu_samples.append(cpu_percent)
            memory_samples.append(memory_mb)
        except psutil.NoSuchProcess:

            # process finished between our check
            break  

    stdout, stderr = process.communicate()

    return stdout, cpu_samples, memory_samples

def summarize_resource_usage(cpu_samples, memory_samples):
    """Summarize CPU and memory usage collected during a pilot run"""

    if not cpu_samples or not memory_samples:
        return None

    return {
        "avg_cpu_percent": sum(cpu_samples) / len(cpu_samples),
        "peak_cpu_percent": max(cpu_samples),
        "avg_memory_mb": sum(memory_samples) / len(memory_samples),
        "peak_memory_mb": max(memory_samples),
    }

def print_pilot_report(dataset_info, averages, bottleneck, resource_summary, estimated_minutes_per_epoch):
    """Print clean, readable summary of a pilot run's findings"""

    print("\nPulse Pilot Run Report")
    print("=" * 50)

    if dataset_info:
        print(f"Dataset size: {dataset_info['dataset_size']}, Batch size: {dataset_info['batch_size']}")

    print(f"\nEstimated time per epoch: {estimated_minutes_per_epoch:.2f} minutes")

    print("\nAverage time per phase (ms):")
    for phase, avg_time in averages.items():
        print(f"  {phase}: {avg_time:.2f}")

    if bottleneck:
        print(f"\nBottleneck detected: {bottleneck['phase']} ({bottleneck['fraction']*100:.1f}% of step time)")
    else:
        print("\nNo dominant bottleneck detected, timing looks fairly balanced.")

    if resource_summary:
        print(f"\nResource usage during pilot run:")
        print(f"  CPU: avg {resource_summary['avg_cpu_percent']:.1f}%, peak {resource_summary['peak_cpu_percent']:.1f}%")
        print(f"  Memory: avg {resource_summary['avg_memory_mb']:.1f} MB, peak {resource_summary['peak_memory_mb']:.1f} MB")
    
def get_gpu_stats():
    """Get current GPU utilization % and memory usage, for one moment in time"""

    # Returns None if no NVIDIA GPU is available (eg: running on a Mac)
    if not GPU_AVAILABLE:
        return None

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return {
        "gpu_percent": utilization.gpu,
        "gpu_memory_used_mb": memory.used / (1024 * 1024),
        "gpu_memory_total_mb": memory.total / (1024 * 1024),
    }

if __name__ == "__main__":
    raw_output, cpu_samples, memory_samples = run_pilot_with_monitoring("examples/toy_cnn_train.py", steps=20)
    dataset_info, steps = parse_pilot_output(raw_output)
    averages = compute_average_timings(steps, warmup_steps=1)

    steps_per_epoch = compute_steps_per_epoch(dataset_info)
    estimated_minutes_per_epoch = estimate_full_run_time(averages, total_steps=steps_per_epoch)
    bottleneck = diagnose_bottleneck(averages)
    resource_summary = summarize_resource_usage(cpu_samples, memory_samples)

    print_pilot_report(dataset_info, averages, bottleneck, resource_summary, estimated_minutes_per_epoch)