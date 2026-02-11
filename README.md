# urban-water-runoff
# Urban Runoff as a Vector of Emerging Contaminants: Chemical fingerprinting via advanced HRMS Workflow and Machine Learning


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

<img width="1328" height="531" alt="graphical abstract" src="https://github.com/user-attachments/assets/47385a94-b3d5-4a32-8849-454ac25e7813" />

This repository contains a machine learning pipeline for classifying urban water runoff samples by their sampling points using high-resolution mass spectrometry (HRMS) nontarget analysis data. The pipeline enables **source apportionment** of runoff water contamination by identifying chemical fingerprints that distinguish different urban drainage locations.

## Study Design

### Sampling Campaign

| Parameter | Details |
|-----------|---------|
| **Location** | Thessaloniki, Central Macedonia, Greece |
| **Sampling Points** | 4 urban locations |
| **Total Samples** | 40 (10 per location) |
| **Sampling Events** | Multiple rain events |
| **Analysis Method** | High-resolution mass spectrometry (HRMS) |

### Sampling Locations

1. **Agias Sofias Str** — Urban commercial district (70% classification accuracy)
2. **Ethnikis Aminis Str** — Residential/commercial mixed zone (60% accuracy)
3. **Lagkada Str** — Industrial/residential interface (80% accuracy)
4. **Thessaloniki Port** — Maritime industrial zone (40% accuracy)

### Compound Identification Levels

Following the Schymanski et al. (2014) framework:

| Dataset | Confidence Level | Compounds | Description |
|---------|------------------|-----------|-------------|
| **Level 1-2** | High confidence | 175 | Confirmed by reference standard or library match |
| **Level 1-3** | Extended | 1,874 | Includes tentative candidates |

---

## Pipeline Architecture

The analysis consists of three integrated components:

### 1. Model Selection 

Systematic evaluation of **16 modeling scenarios** to identify the optimal combination:

- **2 Datasets**: Level 1-2 (175 features) vs Level 1-3 (1,874 features)
- **2 Classification Strategies**: Multi-class vs One-vs-Rest (OvR)
- **4 Machine Learning Models**: Linear SVC, Random Forest, XGBoost, Logistic Regression


### 2. Hyperparameter Optimization

Intensive Bayesian optimization using Optuna (100 trials) to fine-tune the winning model:

| Parameter | Baseline (5 trials) | Optimized (100 trials) |
|-----------|---------------------|------------------------|
| Accuracy | 57.5% | **62.5%** (+8.7%) |
| C (regularization) | 2.47 | 0.026 |
| l1_ratio | 0.60 | 0.14 |

### 3. Validation and Interpretation

Rigorous validation with SHAP-based explainability:

- Out-of-fold predictions using `cross_val_predict`
- Confusion matrix analysis
- SHAP feature importance ranking
- Per-location chemical fingerprint analysis

---



## Output Files

### Figures

| Figure | Description |
|--------|-------------|
| `figure1_dataset_comparison.png` | Boxplot comparing Level 1-2 vs Level 1-3 datasets |
| `figure2_performance_comparison.png` | 4-panel performance metrics by model type |
| `figure3_scenario_heatmap.png` | Heatmap of all 16 modeling scenarios |
| `figure4_top_performers.png` | Top 4 model configurations |
| `figure5_optimization_comparison.png` | Baseline vs optimized model comparison |
| `figure6_optuna_optimization_curve.png` | 100-trial optimization progress |
| `figure7_shap_summary.png` | SHAP beeswarm plot (top 20 features) |
| `figure8a-d_shap_waterfall_*.png` | SHAP waterfall plots for each location |
| `figure9_summary_4panel.png` | Complete analysis summary |

### Data Files

| File | Description |
|------|-------------|
| `model_selection_results.csv` | Results from all 16 scenarios |
| `optimization_summary.csv` | Hyperparameter optimization results |
| `predictions.csv` | Out-of-fold predictions for all samples |
| `feature_importance.csv` | SHAP-ranked feature importance |
| `final_metrics.csv` | Final cross-validation metrics |

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Dependencies

See `requirements.txt` for the complete list of dependencies.

---

## Usage

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/nta-stormwater-classification.git
cd nta-stormwater-classification

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python nta_classification_pipeline.py
```

### Input Data Format

The pipeline expects three tab-separated files:

1. **sample_details.txt** — Sample metadata with columns:
   - `Sample_ID`: Unique sample identifier
   - `Sampling Point`: Location name
   - Additional metadata columns (optional)

2. **CL_1_2a_2b_identifications.txt** — Level 1-2 compound intensities:
   - First column: `Sample` (sample identifier)
   - Remaining columns: Compound names with intensity values

3. **CL_1_2a_2b_3_identifications.txt** — Level 1-3 compound intensities:
   - Same format as Level 1-2

---

## Methodology Summary

### Cross-Validation Strategy

- **Outer CV**: 5-Fold Stratified (model evaluation)
- **Inner CV**: 3-Fold Stratified (hyperparameter tuning)
- **Random State**: 42 (reproducibility)

### Preprocessing

1. Variance threshold filtering (removes constant features)
2. Standard scaling (zero mean, unit variance)

### Hyperparameter Optimization

Bayesian optimization using Optuna with TPE (Tree-structured Parzen Estimator) sampler:

| Model | Parameters | Search Space |
|-------|------------|--------------|
| Linear SVC | C | log[1e-4, 100] |
| Random Forest | max_depth, min_samples_leaf | [2-8], [1-5] |
| XGBoost | max_depth, learning_rate, reg_alpha, reg_lambda | [1-3], [0.01-0.3], log |
| Logistic Regression | C, l1_ratio | log[1e-4, 100], [0-1] |

---

## Citation

```bibtex
@article{YourName2025,
  title={Machine Learning-Based Source Apportionment of Urban Stormwater Runoff 
         Using Nontarget High-Resolution Mass Spectrometry},
  author={Your Name and Collaborators},
  journal={Environmental Science and Technology},
  year={2025},
  doi={XX.XXXX/acs.est.XXXXXXX}
}
```

---

## References

1. Schymanski, E. L., et al. (2014). Identifying small molecules via high resolution mass spectrometry: communicating confidence. *Environmental Science & Technology*, 48(4), 2097-2098.

2. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

3. Akiba, T., et al. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*.

---

## License

MIT License — see LICENSE file for details.

---

## Contact

For questions or collaboration inquiries, please contact [your.email@institution.edu].
