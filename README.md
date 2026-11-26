# Multispectral-Classification
Multispectral Land Cover Classification of Wageningen using Sentinel-2 Images


## Workflow

The project follows these steps:

1. **Data loading and exploration.** Sentinel-2 scenes (12 bands, 18 May 2017) and S2GLC
   land cover labels are loaded for Arnhem, Renkum and Wageningen. Class distributions
   and scene differences are inspected.
2. **Feature engineering.** The 12 spectral bands are extended with spectral indices
   (NDVI, MNDWI, NDBI), computed per scene before flattening each image into a pixel table.
3. **Class imbalance handling.** Class-weighted learning (Random Forest) and class-balanced
   pixel sampling (KNN) are used to avoid ignoring rare classes.
4. **Spatial validation.** Models are trained on Arnhem and validated on Renkum, a
   different area, so the score reflects generalisation to a new location. Spatial block
   cross-validation within Arnhem gives a mean ± standard deviation.
5. **Modelling.** K-Nearest Neighbours (with feature scaling, optionally PCA) and Random
   Forest are compared on identical features and data.
6. **Evaluation.** The main metric is the macro Jaccard index (IoU), the competition metric,
   reported together with per-class IoU and a normalised confusion matrix. Predicted maps
   are compared with the ground truth visually.
