import numpy as np

def augment_light_curve(flux, roll_frac=0.25, noise_std=0.01, scale_min=0.7, scale_max=1.3):
    """
    Applies temporal rolling, Gaussian noise injection, and transit scaling to augment data.
    """
    augmented = [flux.copy()]
    
    # 1. Temporal shift (rolling)
    shift = np.random.randint(1, int(len(flux) * roll_frac) + 1)
    augmented.append(np.roll(flux, shift))
    
    # 2. Gaussian noise injection
    noise = np.random.normal(0, noise_std * np.std(flux), len(flux))
    augmented.append(flux + noise)
    
    # 3. Scale transit depth (assuming transit values have negative dips)
    min_val = np.min(flux)
    if min_val < 0:
        scale_factor = np.random.uniform(scale_min, scale_max)
        augmented.append(flux * scale_factor)
        
    return augmented
