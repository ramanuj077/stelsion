# Developer Research Notes

## Ideas & Theories

### Minor-Axis Attention
By organizing the light curve segment matrix into dimensions (9, 800) where 9 represents parallel feature mappings (or different sections of the phase folding) and 800 is the folded temporal axis, minor-axis attention focuses Spatial Attention across the different phase segments. This helps emphasize sections containing ingress/egress while ignoring flat out-of-transit stellar baselines.

### Multi-Scale Convolutions
InceptionTime convolutions with different kernel lengths (e.g. 3, 5, 9) are expected to capture transits from short-period planets (which have narrow ingress/egress widths) as well as long-period planets (which produce wider dips).

---

## Literature References
1. *AstroNet: A Neural Network for Identifying Exoplanets* (Shallue & Vanderburg, 2018)
2. *InceptionTime: Finding AlexNet for Time Series Classification* (Fawaz et al., 2020)
3. *ExoMiner: A Machine Learning Classifier for Exoplanet Validation* (Valizadegan et al., 2022)
