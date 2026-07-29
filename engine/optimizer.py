from pilot import run_pilot_with_monitoring, parse_pilot_output, compute_average_timings, compute_total_step_time, compute_steps_per_epoch

def get_reliable_step_count(filepath, min_epochs_to_test=1.2):
    """Run a tiny baseline pass just to learn the dataset size, then
    compute a step count that reliably crosses at least 1 epoch boundary"""

    raw_output, cpu_samples, memory_samples, gpu_samples = run_pilot_with_monitoring(filepath, steps=1)
    dataset_info, pilot_steps = parse_pilot_output(raw_output)

    steps_per_epoch = compute_steps_per_epoch(dataset_info)
    return int(steps_per_epoch * min_epochs_to_test)

def test_candidate(filepath, flag_name, flag_value, steps, fixed_args=None):
    """Run the pilot with one specific candidate setting, and return
    its measured average total step time (ms)"""

    extra_args = [flag_name, str(flag_value)]
    if fixed_args:
        extra_args += fixed_args

    raw_output, cpu_samples, memory_samples, gpu_samples = run_pilot_with_monitoring(filepath, steps=steps, extra_args=extra_args)
    dataset_info, pilot_steps = parse_pilot_output(raw_output)
    averages = compute_average_timings(pilot_steps, warmup_steps=1)
    total_step_time = compute_total_step_time(averages)

    return {
        "flag_name": flag_name,
        "flag_value": flag_value,
        "averages": averages,
        "total_step_time_ms": total_step_time
    }

def compare_candidates(filepath, flag_name, candidate_values, steps, fixed_args=None):
    """Test multiple candidate values for one setting, and return
    results sorted from fastest to slowest."""

    results = []
    for value in candidate_values:
        result = test_candidate(filepath, flag_name, value, steps=steps, fixed_args=fixed_args)
        results.append(result)

    results.sort(key=lambda r: r["total_step_time_ms"])
    return results

import time

if __name__ == "__main__":
    steps = get_reliable_step_count("examples/toy_cnn_train.py")
    print(f"Using {steps} steps per candidate test\n")

    t0 = time.time()
    results = compare_candidates(
        "examples/toy_cnn_train.py", "--num-workers", [0, 2, 4, 8],
        steps=steps,
        fixed_args=["--persistent-workers", "true", "--artificial-delay", "0.01"]
    )
    elapsed = time.time() - t0

    for r in results:
        print(f"num_workers={r['flag_value']}: {r['total_step_time_ms']:.2f}ms/step")

    print(f"\nTotal time to test all candidates: {elapsed:.2f}s")