# Lets program run commands or other programs as though you typed them into terminal
# By letting code run another program, we can capture info about what it does
import subprocess

# Gives access to info and controls related to interpreter and program currently running
import sys

# JSON is common text format used to store and send structured data
import json

# Math module has many built-in mathematical functions
import math

def run_pilot(filepath, steps=20):
    """Launch training script as subprocess for limited number of steps"""

    result = subprocess.run(
        [sys.executable, filepath, "--steps", str(steps)],
        capture_output=True,
        text=True
    )

    return result.stdout

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

if __name__ == "__main__":
    raw_output = run_pilot("examples/toy_cnn_train.py", steps=5)
    dataset_info, steps = parse_pilot_output(raw_output)
    averages = compute_average_timings(steps, warmup_steps=1)

    bottleneck = diagnose_bottleneck(averages)
    print("Bottleneck:", bottleneck)