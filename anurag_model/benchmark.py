import sys
import os
import time
import torch

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anurag_model.architecture import UpgradedExoplanetDetectorNet

def benchmark_model():
    print("--- Phase 2 Dual-Input Model Benchmark ---")
    
    # 1. Initialize model
    model = UpgradedExoplanetDetectorNet(input_len=2000, dropout=0.3, num_heads=4)
    model.eval()
    
    # 2. Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")
    
    # 3. Measure inference latency
    # Generate dummy global & local inputs (size 1 for single-sample latency)
    dummy_global = torch.randn(1, 2000)
    dummy_local = torch.randn(1, 200)
    
    # Warmup
    for _ in range(50):
        _ = model(dummy_global, dummy_local)
        
    # Latency loop
    num_runs = 500
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_global, dummy_local)
    end_time = time.perf_counter()
    
    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
    print(f"Average Inference Latency: {avg_latency_ms:.3f} ms")
    
if __name__ == "__main__":
    benchmark_model()
