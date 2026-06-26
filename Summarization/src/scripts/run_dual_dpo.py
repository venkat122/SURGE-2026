import os
import subprocess
import time

def run_command(cmd):
    print(f"Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    if process.returncode != 0:
        print(f"Command failed with code {process.returncode}")
        exit(1)

def main():
    print("=========================================")
    print("=== PHASE 1: Optimized DPO TRAINING SCRIPT ===")
    print("=========================================")
    run_command("conda run --no-capture-output -n legalsum python src/dpo_baseline/train_dpo_baseline.py")
    
    print("\n=========================================")
    print("=== COOLDOWN PHASE: 1 HOUR SLEEP ===")
    print("=========================================")
    print("Optimized DPO finished finished! Pausing execution for 60 minutes to let the laptop GPU cool down...")
    time.sleep(3600)
    
    print("\n=========================================")
    print("=== PHASE 2: Rhetorical DPO TRAINING SCRIPT ===")
    print("=========================================")
    run_command("conda run --no-capture-output -n legalsum python src/dpo_rhetorical/train_dpo_rhetorical.py")
    
    print("\n=========================================")
    print("=== Pipelines finished Finished ===")
    print("=========================================")

if __name__ == "__main__":
    main()
