import time
import subprocess
import sys
import os

print("=========================================")
print("=== COOLDOWN PHASE: 30 MINUTE SLEEP ===")
print("=========================================")
print("Starting 30-minute cooldown to let the GPU rest...")
time.sleep(1800)

print("\n=========================================")
print("=== PHASE 3: INFERENCE BENCHMARKING ===")
print("=========================================")
print("Running evaluate_dpo_models.py...")
result1 = subprocess.run(
    ["conda", "run", "--no-capture-output", "-n", "legalsum", "python", "src/common/evaluate_dpo_models.py"], 
    capture_output=True, 
    text=True,
    cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

eval_output_file = r"C:\Users\samba\.gemini\antigravity\brain\65b10c10-535e-480c-9673-8ae8ea226dea\dpo_eval_results.md"
with open(eval_output_file, "w") as f:
    f.write("# DPO Evaluation Results\n\n```text\n")
    f.write(result1.stdout)
    if result1.stderr:
        f.write("\nErrors:\n" + result1.stderr)
    f.write("\n```\n")
print(result1.stdout)

print("\n=========================================")
print("=== PHASE 4: DYNAMIC CHUNKING STRESS TEST ===")
print("=========================================")
print("Running run_chunking_on_all.py...")
result2 = subprocess.run(
    ["conda", "run", "--no-capture-output", "-n", "legalsum", "python", "src/dynamic_chunking/run_chunking_on_all.py"], 
    capture_output=True, 
    text=True,
    cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)
print(result2.stdout)

print("\n=========================================")
print("=== Tasks finished Finished ===")
print("=========================================")
