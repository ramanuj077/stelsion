import os
import sys
import argparse
import subprocess
import ramanuj_model.config as config
from ramanuj_model.trainer import train_model, run_optuna_study

def main():
    parser = argparse.ArgumentParser(description="STELSION Personal Research Workspace: Training and HPO CLI")
    parser.add_argument(
        "--optuna", 
        action="store_true", 
        help="Run hyperparameter search via Optuna instead of a single training run"
    )
    args = parser.parse_args()
    
    if args.optuna:
        print("Starting hyperparameter optimization study using Optuna...")
        best_params = run_optuna_study()
        print("Best parameters found:")
        print(best_params)
        
        # Optionally, train a final model with the best parameters
        print("\nTraining final model with best parameters...")
        model_path, metrics = train_model(config_overrides=best_params)
    else:
        print(f"Starting standard training run using architecture: {config.ARCHITECTURE}...")
        model_path, metrics = train_model()
        
    print(f"\nTraining session finished. Final model saved to: {model_path}")
    
    # Check if a test dataset exists for auto-benchmarking
    test_dir = "datasets/test"
    test_files_exist = False
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.endswith(".json") or f.endswith(".csv") or f.endswith(".npy")]
        if test_files:
            test_files_exist = True
            
    if test_files_exist:
        print("\n" + "=" * 50)
        print("RUNNING AUTO-BENCHMARK...")
        print("=" * 50)
        # Execute evaluation/benchmark.py to record the model performance in leaderboard.csv
        benchmark_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation", "benchmark.py"))
        try:
            # We run it using the same python interpreter (sys.executable)
            subprocess.run([sys.executable, benchmark_script, model_path], check=True)
            print("Auto-benchmark logging completed successfully!")
        except Exception as e:
            print(f"Warning: Failed to run auto-benchmark: {e}")
    else:
        print("\n[Notice] No test dataset found in datasets/test/. Skipping auto-benchmarking.")
        print("Place test light curves in datasets/test/ to enable auto-benchmarking upon training completion.")

if __name__ == "__main__":
    main()
