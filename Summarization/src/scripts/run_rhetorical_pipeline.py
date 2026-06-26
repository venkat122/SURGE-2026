import os
import subprocess
import sys

def run_command(cmd, step_name):
    print(f"\n{'='*60}")
    print(f"=== {step_name.upper()} ===")
    print(f"{'='*60}")
    print(f"Executing: {cmd}")
    
    process = subprocess.Popen(cmd, shell=True)
    process.communicate()
    
    if process.returncode != 0:
        print(f"\nError: Pipeline aborted at step: {step_name}")
        sys.exit(1)
    
    print(f"[SUCCESS] Completed step: {step_name}")

def main():
    print("""
============================================================
       SURGE-2026: TRUE END-TO-END RHETORICAL PIPELINE
============================================================

This script chains together the full rhetorical pipeline:
  1. Rhetorical Tagging (Semantic Parsing via OpenNYAI)
  2. Supervised Fine-Tuning (SFT) of the Rhetorical Model
  3. Direct Preference Optimization (DPO) of the Rhetorical Model

Note: Due to hardware constraints, this unified script was modularized 
during active development, but is provided here as the definitive 
end-to-end production pipeline.
""")

    # Step 1: Tag the raw dataset using OpenNYAI
    run_command(
        cmd="conda run --no-capture-output -n legalsum python src/rhetorical_role/run_opennyai_local.py",
        step_name="Phase 1: Semantic Rhetorical Tagging"
    )

    # Step 2: SFT on the tagged dataset
    run_command(
        cmd="conda run --no-capture-output -n legalsum python src/rhetorical_role/train_rhetorical_role.py",
        step_name="Phase 2: Supervised Fine-Tuning (Rhetorical Model)"
    )

    # Step 3: DPO Alignment
    run_command(
        cmd="conda run --no-capture-output -n legalsum python src/dpo_rhetorical/train_dpo_rhetorical.py",
        step_name="Phase 3: Direct Preference Optimization (Rhetorical DPO)"
    )

    print("\n" + "="*60)
    print("=== PIPELINE Finished. ===")
    print("The final models have been finished trained and aligned.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
