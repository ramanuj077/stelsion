import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.filters import (
    handle_missing_values, normalize_flux, remove_outliers_sigma_clipping,
    wavelet_denoising, savitzky_golay_filter, median_filter,
    remove_stellar_variability, estimate_noise, hybrid_denoising
)

class PreprocessingPipeline:
    def __init__(self, config=None):
        self.config = config or {
            'missing_value_method': 'interpolate',
            'normalization_method': 'median',
            'sigma_clipping_sigma': 3.0,
            'sigma_clipping_iters': 2,
            'wavelet_type': 'db4',
            'wavelet_level': 2,
            'sg_window': 15,
            'sg_polyorder': 2,
            'median_kernel': 5,
            'stellar_var_window': 101,
            'segment_length': 2000,
            'enable_augmentation': True,
            'test_size': 0.2,
            'val_size': 0.1,
            'random_state': 42
        }

    def process_single_curve(self, flux, steps=None):
        """
        Runs the specified preprocessing steps on a single light curve,
        using an Adaptive Noise Analyzer to select the denoising method.
        """
        if steps is None:
            steps = ['missing_values', 'sigma_clip', 'stellar_var', 'denoise', 'normalize']
            
        flux = np.array(flux, dtype=float)
        
        for step in steps:
            if step == 'missing_values':
                flux = handle_missing_values(flux, method=self.config.get('missing_value_method', 'interpolate'))
            elif step == 'sigma_clip':
                flux = remove_outliers_sigma_clipping(
                    flux, 
                    sigma=self.config.get('sigma_clipping_sigma', 3.0),
                    iters=self.config.get('sigma_clipping_iters', 2)
                )
            elif step == 'stellar_var':
                flux = remove_stellar_variability(
                    flux,
                    window_size=self.config.get('stellar_var_window', 101)
                )
            elif step == 'denoise':
                # Adaptive Noise Analyzer
                noise = estimate_noise(flux)
                if noise < 0.005:
                    # Low Noise -> Savitzky-Golay only
                    flux = savitzky_golay_filter(
                        flux,
                        window_size=self.config.get('sg_window', 15),
                        polyorder=self.config.get('sg_polyorder', 2)
                    )
                elif noise < 0.015:
                    # Medium Noise -> Wavelet denoising
                    flux = wavelet_denoising(
                        flux,
                        wavelet=self.config.get('wavelet_type', 'db4'),
                        level=self.config.get('wavelet_level', 2)
                    )
                else:
                    # High Noise -> Hybrid Wavelet + Median + Sigma clipping
                    flux = hybrid_denoising(
                        flux,
                        wavelet=self.config.get('wavelet_type', 'db4'),
                        level=self.config.get('wavelet_level', 2),
                        sg_window=self.config.get('sg_window', 15),
                        sg_polyorder=self.config.get('sg_polyorder', 2),
                        median_kernel=self.config.get('median_kernel', 5),
                        sigma=self.config.get('sigma_clipping_sigma', 3.0),
                        iters=self.config.get('sigma_clipping_iters', 2)
                    )
            elif step == 'normalize':
                flux = normalize_flux(flux, method=self.config.get('normalization_method', 'median'))
                
        return flux

    def augment_light_curve(self, flux):
        """
        Applies random roll (temporal shift), Gaussian noise injection, and scaling to augment data.
        """
        augmented = []
        # 1. Original
        augmented.append(flux)
        
        if not self.config.get('enable_augmentation', True):
            return augmented

        # 2. Roll/Shift
        shift = np.random.randint(1, len(flux) // 4)
        augmented.append(np.roll(flux, shift))
        
        # 3. Noise Injection
        noise = np.random.normal(0, 0.01 * np.std(flux), len(flux))
        augmented.append(flux + noise)
        
        # 4. Scale transit depth (assuming transit signal is negative dips)
        min_val = np.min(flux)
        if min_val < 0:
            scale_factor = np.random.uniform(0.7, 1.3)
            scaled = flux * scale_factor
            augmented.append(scaled)
            
        return augmented

    def segment_or_pad(self, flux, target_len):
        """
        Pads or crops the light curve to a fixed target length.
        """
        curr_len = len(flux)
        if curr_len == target_len:
            return flux
        elif curr_len > target_len:
            # Crop middle segment
            start = (curr_len - target_len) // 2
            return flux[start:start+target_len]
        else:
            # Pad with edge value or zeros
            pad_width = target_len - curr_len
            return np.pad(flux, (0, pad_width), mode='edge')

    def prepare_dataset(self, x_data, y_data):
        """
        Preprocesses, segments, balances, and splits the dataset.
        """
        processed_x = []
        processed_y = []
        
        target_len = self.config.get('segment_length', 2000)
        
        for curve, label in zip(x_data, y_data):
            clean_curve = self.process_single_curve(curve)
            clean_curve = self.segment_or_pad(clean_curve, target_len)
            processed_x.append(clean_curve)
            processed_y.append(label)
                
        X = np.array(processed_x, dtype=np.float32)
        y = np.array(processed_y, dtype=np.float32)
        
        # Split: Train, Validation, Test
        test_size = self.config.get('test_size', 0.2)
        val_size = self.config.get('val_size', 0.1)
        r_state = self.config.get('random_state', 42)
        
        # First split off test set (stratified by y to preserve proportions)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=r_state, stratify=y
        )
        
        # Calculate validation size relative to the remaining train_val set
        rel_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=rel_val_size, random_state=r_state, stratify=y_train_val
        )
        
        # Apply data augmentation ONLY to the training set
        X_train_aug = []
        y_train_aug = []
        
        for curve, label in zip(X_train, y_train):
            if label == 1 and self.config.get('enable_augmentation', True):
                aug_curves = self.augment_light_curve(curve)
                for ac in aug_curves:
                    X_train_aug.append(ac)
                    y_train_aug.append(1)
            else:
                X_train_aug.append(curve)
                y_train_aug.append(label)
                
        X_train_aug = np.array(X_train_aug, dtype=np.float32)
        y_train_aug = np.array(y_train_aug, dtype=np.float32)
        
        # Balance the training set (1:1 ratio)
        c0_idx = np.where(y_train_aug == 0)[0]
        c1_idx = np.where(y_train_aug == 1)[0]
        
        n0 = len(c0_idx)
        n1 = len(c1_idx)
        
        if n0 > 0 and n1 > 0:
            target_size = max(n0, n1)
            
            # Oversample class 0 if needed
            if n0 < target_size:
                np.random.seed(r_state)
                extra_idx = np.random.choice(c0_idx, size=target_size - n0, replace=True)
                c0_balanced_idx = np.concatenate([c0_idx, extra_idx])
            else:
                c0_balanced_idx = c0_idx
                
            # Oversample class 1 if needed
            if n1 < target_size:
                np.random.seed(r_state)
                extra_idx = np.random.choice(c1_idx, size=target_size - n1, replace=True)
                c1_balanced_idx = np.concatenate([c1_idx, extra_idx])
            else:
                c1_balanced_idx = c1_idx
                
            balanced_idx = np.concatenate([c0_balanced_idx, c1_balanced_idx])
            # Shuffle the balanced training dataset
            np.random.seed(r_state)
            np.random.shuffle(balanced_idx)
            
            X_train_balanced = X_train_aug[balanced_idx]
            y_train_balanced = y_train_aug[balanced_idx]
        else:
            # Fallback if one class has 0 samples in train split
            X_train_balanced = X_train_aug
            y_train_balanced = y_train_aug
            
        return {
            'train': (X_train_balanced, y_train_balanced),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
