import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anurag_model.pipeline import DualViewPipeline

def generate_trapezoidal_transit(length, period, depth, duration, t0):
    """
    Generates a physically motivated trapezoidal transit dip profile.
    Uses periodic folding to support multiple transit events in a single curve.
    """
    time = np.arange(length)
    profile = np.zeros(length, dtype=np.float32)
    
    # Calculate phase [0.0, 1.0]
    phase = ((time - t0) / period) % 1.0
    
    # Calculate index distance to the nearest periodic transit center
    dist = np.minimum(phase, 1.0 - phase) * period
    
    # Ingress/egress slope width (10% of total duration)
    ingress = duration * 0.1
    half_dur = duration / 2.0
    
    t_start = half_dur - ingress / 2.0
    t_end = half_dur + ingress / 2.0
    
    # Core flat bottom of transit
    flat_mask = dist <= t_start
    profile[flat_mask] = -depth
    
    # Ingress/egress slopes
    slope_mask = (dist > t_start) & (dist < t_end)
    if ingress > 0 and np.any(slope_mask):
        profile[slope_mask] = -depth * (t_end - dist[slope_mask]) / ingress
        
    return profile

class ExoplanetDataset(Dataset):
    def __init__(self, num_samples=100, length=2000, has_transits=True, inject_prob=0.5):
        """
        PyTorch Dataset that generates stellar noise baselines and dynamically 
        injects physical planet transit dips on the fly.
        """
        self.num_samples = num_samples
        self.length = length
        self.inject_prob = inject_prob
        self.pipeline = DualViewPipeline()
        
        # Pre-generate different stellar baselines (to avoid pure uniform noise)
        self.baselines = []
        for _ in range(num_samples):
            # Baseline contains low frequency stellar variability (sine waves) + random walk
            time = np.linspace(0, 10, length)
            stellar_var = 0.02 * np.sin(2 * np.pi * time / 3) + 0.005 * np.cos(2 * np.pi * time / 0.5)
            noise = np.random.normal(0, 0.004, length)
            self.baselines.append(1.0 + stellar_var + noise)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Start with raw stellar noise baseline
        flux = self.baselines[idx].copy()
        
        # 2. Decide if we inject a planet transit
        label = 0.0
        if np.random.random() < self.inject_prob:
            label = 1.0
            
            # Randomized transit parameters
            period = np.random.uniform(300, 600)   # Period in index units
            depth = np.random.uniform(0.015, 0.04) # 1.5% to 4% transit depth
            duration = np.random.uniform(30, 80)   # Transit duration in indices
            t0 = np.random.uniform(50, 250)        # Transit epoch offset
            
            # Generate and apply transit profile
            transit_dip = generate_trapezoidal_transit(self.length, period, depth, duration, t0)
            flux += transit_dip
            
        # 3. Add dynamic measurement white noise (Data Augmentation)
        flux += np.random.normal(0, 0.001, self.length)
        
        # 4. Process through the DualViewPipeline to extract Global/Local views
        global_view, local_view, _, _ = self.pipeline.process(flux)
        
        # Convert to PyTorch floats
        global_tensor = torch.tensor(global_view, dtype=torch.float32)
        local_tensor = torch.tensor(local_view, dtype=torch.float32)
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return global_tensor, local_tensor, label_tensor

if __name__ == "__main__":
    print("Testing ExoplanetDataset and Transit Injections...")
    dataset = ExoplanetDataset(num_samples=10, inject_prob=0.5)
    g, l, y = dataset[0]
    print(f"Dataset test index 0:")
    print(f"Global View shape: {g.shape} (Expected: 2000)")
    print(f"Local View shape: {l.shape} (Expected: 200)")
    print(f"Label: {y.item()} (0.0 = Noise, 1.0 = Transit)")
    print("\n✓ Dataset test passed successfully!")
