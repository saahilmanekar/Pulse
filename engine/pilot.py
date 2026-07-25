# Lets program run commands or other programs as though you typed them into terminal
# By letting code run another program, we can capture info about what it does
import subprocess

# Gives access to info and controls related to interpreter and program currently running
import sys

def run_pilot(filepath, steps=20):
    """Launch training script as subprocess for limited number of steps"""

    result = subprocess.run(
        [sys.executable, filepath, "--steps", str(steps)],
        capture_output=True,
        text=True
    )

    return result.stdout

if __name__ == "__main__":
    result = subprocess.run(
        ["python3", "examples/toy_cnn_train.py", "--steps", "5"],
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)