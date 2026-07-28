from checker import run_static_check
from pilot import run_pilot_with_monitoring, parse_pilot_output, compute_average_timings, diagnose_bottleneck

def gather_findings(filepath, steps=20):
    """Run both the static checker and a pilot run on the same script
    and return their raw findings together"""

    static_findings = run_static_check(filepath)

    raw_output, cpu_samples, memory_samples, gpu_samples = run_pilot_with_monitoring(filepath, steps=steps)
    dataset_info, pilot_steps = parse_pilot_output(raw_output)
    averages = compute_average_timings(pilot_steps, warmup_steps=1)
    bottleneck = diagnose_bottleneck(averages)

    return {
        "static_findings": static_findings,
        "bottleneck": bottleneck,
        "averages": averages,
    }

if __name__ == "__main__":
    results = gather_findings("examples/toy_cnn_train.py")
    print(results)