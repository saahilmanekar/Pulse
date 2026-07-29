from pilot import run_pilot_with_monitoring, parse_pilot_output, compute_average_timings, compute_total_step_time

def test_candidate(filepath, flag_name, flag_value, steps=20):
    """Run the pilot with one specific candidate setting, and return
    its measured average total step time (ms)"""

    extra_args = [flag_name, str(flag_value)]

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