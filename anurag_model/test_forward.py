import sys
import os
import torch

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anurag_model.architecture import UpgradedExoplanetDetectorNet

def run_forward_pass_test():
    print("--- Starting Phase 2 Dual-Input Forward Pass Validation ---")
    
    # 1. Initialize the upgraded model
    print("Initializing UpgradedExoplanetDetectorNet...")
    model = UpgradedExoplanetDetectorNet(input_len=2000, dropout=0.3, num_heads=4)
    model.eval()
    print("Model initialized successfully.")
    
    # 2. Generate dummy dual-input tensors
    batch_size = 4
    global_len = 2000
    local_len = 200
    print(f"Creating dummy global input of shape (Batch={batch_size}, Len={global_len})...")
    dummy_global = torch.randn(batch_size, global_len)
    
    print(f"Creating dummy local input of shape (Batch={batch_size}, Len={local_len})...")
    dummy_local = torch.randn(batch_size, local_len)
    
    # 3. Perform forward pass
    print("Running forward pass with dual inputs...")
    with torch.no_grad():
        preds, attn = model(dummy_global, dummy_local)
        
    print("Forward pass completed successfully.")
    
    # 4. Check shapes and bounds
    print(f"Prediction output shape: {preds.shape} (Expected: ({batch_size}, 1))")
    print(f"Attention matrix shape: {attn.shape} (Expected: ({batch_size}, 63, 63))")
    
    # Verify predictions are probabilities [0, 1]
    min_pred = preds.min().item()
    max_pred = preds.max().item()
    print(f"Output probability range: [{min_pred:.4f}, {max_pred:.4f}]")
    
    # Assertions
    assert preds.shape == (batch_size, 1), f"Incorrect prediction shape: {preds.shape}"
    assert attn.shape == (batch_size, 63, 63), f"Incorrect attention shape: {attn.shape}"
    assert 0.0 <= min_pred <= 1.0, f"Probability out of range: {min_pred}"
    assert 0.0 <= max_pred <= 1.0, f"Probability out of range: {max_pred}"
    
    print("\n✓ Verification Successful! Dual-input model compiles and runs with correct shape matching.")

if __name__ == "__main__":
    try:
        run_forward_pass_test()
    except Exception as e:
        print(f"\n✗ Test Failed: {str(e)}")
        sys.exit(1)
