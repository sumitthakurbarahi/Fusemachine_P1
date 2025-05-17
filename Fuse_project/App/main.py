import os
import subprocess

def run_scripts_in_order():
    scripts = [
        "notebooks/fourbar_animator.py",
        "notebooks/optimizer.py",
    ]

    base_path = os.path.dirname(os.path.dirname(__file__))  # Fuse_project directory
    curve_csv_path = os.path.join(base_path, "notebooks", "curve.csv")

    for script in scripts:
        if script == "notebooks/optimizer.py" and not os.path.exists(curve_csv_path):
            print(f"Dependency missing: {curve_csv_path} not found. Skipping {script}.\n")
            continue

        script_path = os.path.join(base_path, script)
        if os.path.exists(script_path):
            print(f"Running {script}...")
            result = subprocess.run(["python3", script_path], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully ran {script}.\n")
            else:
                print(f"Error running {script}:\n{result.stderr}\n")
                break
        else:
            print(f"Script {script} not found. Skipping...\n")