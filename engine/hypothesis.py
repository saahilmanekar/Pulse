from checker import run_static_check
from pilot import run_pilot_with_monitoring, parse_pilot_output, compute_average_timings, diagnose_bottleneck

def gather_findings(filepath, steps=20, extra_args=None):
    """Run both the static checker and a pilot run on the same script
    and return their raw findings together"""

    # Gather static findings
    static_findings = run_static_check(filepath)

    # Gather pilot findings
    raw_output, cpu_samples, memory_samples, gpu_samples = run_pilot_with_monitoring(filepath, steps=steps, extra_args=extra_args)
    dataset_info, pilot_steps = parse_pilot_output(raw_output)
    averages = compute_average_timings(pilot_steps, warmup_steps=1)
    bottleneck = diagnose_bottleneck(averages)

    return {
        "static_findings": static_findings,
        "bottleneck": bottleneck,
        "averages": averages,
    }

# Map each bottleneck phase to keywords that would appear in a related static finding's message
PHASE_RELATED_KEYWORDS = {
    "data_loading_ms": ["num_workers", "pin_memory", "persistent_workers"],
    "transfer_ms": ["non_blocking", "pin_memory"],
    "forward_ms": ["autocast", "GradScaler"],
    "backward_ms": ["autocast", "GradScaler"],
    "optimizer_ms": [],
}

def build_hypothesis(static_findings, bottleneck):
    """Check whether any static finding relates to the pilot run's
    diagnosed bottleneck phase. Returns a combined, honest hypothesis."""

    if bottleneck is None:
        return {
            "confidence": "low",
            "summary": "No dominant bottleneck was measured, and static findings exist independently. No strong combined hypothesis."
        }

    # Find specific bottleneck phase
    phase = bottleneck["phase"]
    related_keywords = PHASE_RELATED_KEYWORDS.get(phase, [])

    # See if there are any matching findings in static_findings
    matching_findings = []
    for finding in static_findings:
        for keyword in related_keywords:
            if keyword in finding["message"]:
                matching_findings.append(finding)
                break

    if matching_findings:
        return {
            "confidence": "high",
            "summary": f"{phase} is the measured bottleneck ({bottleneck['fraction']*100:.1f}% of step time), AND the static checker found related code issues, these two independent signals reinforce each other.",
            "related_findings": matching_findings,
        }
    else:
        return {
            "confidence": "medium",
            "summary": f"{phase} is the measured bottleneck ({bottleneck['fraction']*100:.1f}% of step time), but no related static findings were found. This may be a compute-bound limitation (eg: backward pass) rather than a configuration issue Pulse can currently detect.",
        }