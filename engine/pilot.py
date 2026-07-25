# Lets program run commands or other programs as though you typed them into terminal
# By letting code run another program, we can capture info about what it does
import subprocess

# Gives access to info and controls related to interpreter and program currently running
import sys

# JSON is common text format used to store and send structured data
import json

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
            step_data = json.loads(line)
            steps.append(step_data)

    return steps

if __name__ == "__main__":
    raw_output = run_pilot("examples/toy_cnn_train.py", steps=5)
    steps = parse_pilot_output(raw_output)
    print(steps)
    print(f"\nNumber of steps parsed: {len(steps)}")
    print(f"First step's data_loading_ms: {steps[0]['data_loading_ms']}")