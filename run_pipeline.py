# run_pipeline.py

import argparse
import subprocess
import sys

def run(step_name, module):
    print(f"\n=== RUNNING: {step_name} ===")
    result = subprocess.run([sys.executable, "-m", module])
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed!")
    print(f"=== DONE: {step_name} ===\n")


def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline steps.")

    # Individual steps
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--train", action="store_true", help="Run final training")
    parser.add_argument("--infer", action="store_true", help="Run inference on test set")
    parser.add_argument("--error", action="store_true", help="Run error analysis")
    parser.add_argument("--viz", action="store_true", help="Generate visualizations")

    # Full pipeline
    parser.add_argument("--all", action="store_true", help="Run full pipeline in correct order")

    args = parser.parse_args()

    # If --all is used, enforce correct order
    if args.all:
        args.preprocess = True
        args.tune = True
        args.train = True
        args.infer = True
        args.error = True
        args.viz = True

    # Execute steps in correct order
    if args.preprocess:
        run("Preprocessing", "scripts.run_preprocessing")

    if args.tune:
        run("Optuna Hyperparameter Tuning", "scripts.run_optuna")

    if args.train:
        run("Final Training", "scripts.run_final_training")

    if args.infer:
        run("Inference", "scripts.run_inference")

    if args.error:
        run("Error Analysis", "scripts.run_error_analysis")

    if args.viz:
        run("Visualizations", "scripts.run_visualizations")

    # If no flags provided
    if not any(vars(args).values()):
        print("No flags provided. Use --help to see available options.")


if __name__ == "__main__":
    main()
