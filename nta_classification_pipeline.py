#!/usr/bin/env python3
"""
================================================================================
NTA STORMWATER CLASSIFICATION PIPELINE
================================================================================

A complete machine learning pipeline for Nontarget Analysis (NTA) of stormwater
runoff samples from urban drainage systems. This pipeline performs:

1. MODEL SELECTION: Evaluates 16 modeling scenarios (2 datasets × 2 strategies × 4 models)
2. HYPERPARAMETER OPTIMIZATION: Intensive Optuna tuning (100 trials) on the best model
3. VALIDATION & INTERPRETATION: Cross-validation metrics, confusion matrix, and SHAP analysis

Author: [Your Name]
Date: 2025
Dataset: Thessaloniki Urban Stormwater (N=40, 4 locations)

================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from io import StringIO
import time
from datetime import datetime
import argparse
import os

# Scikit-learn imports
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score, 
                             recall_score, confusion_matrix, classification_report)

# XGBoost
from xgboost import XGBClassifier

# Optuna for hyperparameter tuning
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# SHAP for explainability
import shap

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Cross-validation settings
N_FOLDS_OUTER = 5
N_FOLDS_INNER = 3
N_TRIALS_MODEL_SELECTION = 5
N_TRIALS_OPTIMIZATION = 100
N_JOBS = -1

# Output directory (change this to your preferred location)
OUTPUT_DIR = './outputs'

# Color palette for publication
COLORS = {
    'level_1_2': '#1B5E20',      # Dark Green
    'level_1_3': '#4A148C',      # Deep Purple
    'multiclass': '#01579B',     # Dark Blue
    'ovr': '#006064',            # Dark Teal
    'baseline': '#1B5E20',       # Dark Green
    'optimized': '#4A148C',      # Deep Purple
    'random_guess': '#D32F2F',   # Red
    'highlight': '#FF8F00',      # Gold
}

# Publication style settings
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_sample_details(filepath='sample_details.txt'):
    """Load sample details metadata."""
    return pd.read_csv(filepath, sep='\t', encoding='latin-1')

def load_identifications(filepath):
    """Load compound identification data."""
    return pd.read_csv(filepath, sep='\t', encoding='latin-1')

# =============================================================================
# PART 1: MODEL SELECTION
# =============================================================================

def run_model_selection(sample_details, df_level_1_2, df_level_1_3, output_dir):
    """
    Evaluate 16 modeling scenarios to find the best combination of:
    - Dataset (Level 1-2 vs Level 1-3)
    - Strategy (MultiClass vs One-vs-Rest)
    - Model (LinearSVC, RandomForest, XGBoost, LogisticRegression)
    """
    
    print("=" * 80)
    print("PART 1: MODEL SELECTION")
    print("=" * 80)
    print(f"Evaluating 16 scenarios (2 datasets × 2 strategies × 4 models)")
    print()
    
    # Prepare datasets
    le = LabelEncoder()
    
    datasets = {
        'Level 1-2': df_level_1_2,
        'Level 1-3': df_level_1_3
    }
    
    strategies = ['MultiClass', 'OvR']

    # Define model configurations
    # Note: XGBoost needs different config for MultiClass vs OvR
    def get_model_configs(strategy, n_classes):
        """Get model configurations appropriate for the strategy."""
        
        configs = {
            'LinearSVC': {
                'model_class': LinearSVC,
                'base_params': {'class_weight': 'balanced', 'max_iter': 5000, 
                               'random_state': RANDOM_STATE, 'dual': False},
                'tune_params': lambda trial: {'C': trial.suggest_float('C', 1e-4, 100, log=True)}
            },
            'RandomForest': {
                'model_class': RandomForestClassifier,
                'base_params': {'class_weight': 'balanced', 'random_state': RANDOM_STATE, 
                               'n_jobs': N_JOBS, 'n_estimators': 100},
                'tune_params': lambda trial: {
                    'max_depth': trial.suggest_int('max_depth', 2, 8),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                }
            },
            'LogisticRegression': {
                'model_class': LogisticRegression,
                'base_params': {'penalty': 'elasticnet', 'solver': 'saga', 
                               'class_weight': 'balanced', 'max_iter': 2000,
                               'random_state': RANDOM_STATE, 'n_jobs': N_JOBS},
                'tune_params': lambda trial: {
                    'C': trial.suggest_float('C', 1e-4, 100, log=True),
                    'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)
                }
            }
        }
        
        # XGBoost configuration depends on strategy
        if strategy == 'MultiClass':
            # Multi-class: use multi:softmax with num_class
            configs['XGBoost'] = {
                'model_class': XGBClassifier,
                'base_params': {
                    'objective': 'multi:softmax',
                    'num_class': n_classes,
                    'eval_metric': 'mlogloss',
                    'use_label_encoder': False,
                    'random_state': RANDOM_STATE,
                    'n_jobs': N_JOBS,
                    'n_estimators': 100,
                    'verbosity': 0
                },
                'tune_params': lambda trial: {
                    'max_depth': trial.suggest_int('max_depth', 1, 3),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True)
                }
            }
        else:
            # OvR: use binary:logistic (will be wrapped in OneVsRestClassifier)
            configs['XGBoost'] = {
                'model_class': XGBClassifier,
                'base_params': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'random_state': RANDOM_STATE,
                    'n_jobs': N_JOBS,
                    'n_estimators': 100,
                    'verbosity': 0
                },
                'tune_params': lambda trial: {
                    'max_depth': trial.suggest_int('max_depth', 1, 3),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True)
                }
            }
        
        return configs
    
    results = []
    
    for dataset_name, df in datasets.items():
        # Merge with sample details
        merged = df.merge(sample_details, left_on='Sample', right_on='Sample_ID')
        feature_cols = [c for c in df.columns if c != 'Sample']
        X_raw = merged[feature_cols].values.astype(float)
        y = le.fit_transform(merged['Sampling Point'].values)
        n_classes_actual = len(np.unique(y))
        
        # Preprocessing
        var_thresh = VarianceThreshold(threshold=0)
        X_var = var_thresh.fit_transform(X_raw)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_var)
        
        n_features = X_scaled.shape[1]
        print(f"Dataset: {dataset_name} ({n_features} features)")
        
        for strategy in strategies:
            # Get model configs appropriate for this strategy
            model_configs = get_model_configs(strategy, n_classes_actual)
            
            for model_name, config in model_configs.items():
                print(f"  Testing: {strategy} + {model_name}...", end=" ", flush=True)
                
                cv_outer = StratifiedKFold(n_splits=N_FOLDS_OUTER, shuffle=True, 
                                           random_state=RANDOM_STATE)
                cv_inner = StratifiedKFold(n_splits=N_FOLDS_INNER, shuffle=True, 
                                           random_state=RANDOM_STATE)
                
                fold_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
                
                try:
                    for train_idx, val_idx in cv_outer.split(X_scaled, y):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        # Inner CV for hyperparameter tuning
                        def objective(trial):
                            params = {**config['base_params'], **config['tune_params'](trial)}
                            model = config['model_class'](**params)
                            if strategy == 'OvR' and model_name != 'LogisticRegression':
                                model = OneVsRestClassifier(model)
                            scores = cross_val_score(model, X_train, y_train, cv=cv_inner,
                                                    scoring='balanced_accuracy', n_jobs=N_JOBS)
                            return np.mean(scores)
                        
                        study = optuna.create_study(direction='maximize',
                                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
                        study.optimize(objective, n_trials=N_TRIALS_MODEL_SELECTION, show_progress_bar=False)
                        
                        # Train with best params
                        best_params = {**config['base_params'], **study.best_params}
                        model = config['model_class'](**best_params)
                        if strategy == 'OvR' and model_name != 'LogisticRegression':
                            model = OneVsRestClassifier(model)
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        
                        fold_metrics['accuracy'].append(balanced_accuracy_score(y_val, y_pred))
                        fold_metrics['f1'].append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
                        fold_metrics['precision'].append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
                        fold_metrics['recall'].append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
                    
                    results.append({
                        'Dataset': dataset_name,
                        'Strategy': strategy,
                        'Model': model_name,
                        'N_Features': n_features,
                        'Accuracy_Mean': np.mean(fold_metrics['accuracy']),
                        'Accuracy_Std': np.std(fold_metrics['accuracy']),
                        'F1_Mean': np.mean(fold_metrics['f1']),
                        'F1_Std': np.std(fold_metrics['f1']),
                        'Precision_Mean': np.mean(fold_metrics['precision']),
                        'Precision_Std': np.std(fold_metrics['precision']),
                        'Recall_Mean': np.mean(fold_metrics['recall']),
                        'Recall_Std': np.std(fold_metrics['recall'])
                    })
                    
                    print(f"Accuracy: {results[-1]['Accuracy_Mean']:.3f}")
                    
                except Exception as e:
                    print(f"FAILED: {str(e)[:50]}")
                    results.append({
                        'Dataset': dataset_name,
                        'Strategy': strategy,
                        'Model': model_name,
                        'N_Features': n_features,
                        'Accuracy_Mean': 0.25,
                        'Accuracy_Std': 0.0,
                        'F1_Mean': 0.25,
                        'F1_Std': 0.0,
                        'Precision_Mean': 0.25,
                        'Precision_Std': 0.0,
                        'Recall_Mean': 0.25,
                        'Recall_Std': 0.0
                    })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'model_selection_results.csv'), index=False)
    
    # Find best model
    best_idx = results_df['Accuracy_Mean'].idxmax()
    best = results_df.loc[best_idx]
    
    print()
    print(f"BEST MODEL: {best['Dataset']} + {best['Strategy']} + {best['Model']}")
    print(f"  Accuracy: {best['Accuracy_Mean']:.4f} (±{best['Accuracy_Std']:.4f})")
    print()
    
    # Generate figures
    generate_model_selection_figures(results_df, output_dir)
    
    return results_df, best

def generate_model_selection_figures(results_df, output_dir):
    """Generate model selection visualization figures."""
    
    models = ['LinearSVC', 'LogisticRegression', 'RandomForest', 'XGBoost']
    
    # =========================================================================
    # Figure 1: Dataset Comparison - Grouped bar chart by model
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(12, 7), facecolor='white')
    
    x = np.arange(len(models))
    width = 0.35
    
    # Get mean and std for each model and dataset
    l12_means = []
    l12_stds = []
    l13_means = []
    l13_stds = []
    
    for model in models:
        l12_data = results_df[(results_df['Dataset'] == 'Level 1-2') & (results_df['Model'] == model)]
        l13_data = results_df[(results_df['Dataset'] == 'Level 1-3') & (results_df['Model'] == model)]
        
        l12_means.append(l12_data['Accuracy_Mean'].mean())
        l12_stds.append(l12_data['Accuracy_Std'].mean())
        l13_means.append(l13_data['Accuracy_Mean'].mean())
        l13_stds.append(l13_data['Accuracy_Std'].mean())
    
    bars1 = ax1.bar(x - width/2, l12_means, width, yerr=l12_stds, 
                    label='Levels 1-2', color=COLORS['level_1_2'], 
                    edgecolor='black', linewidth=1.5, capsize=5)
    bars2 = ax1.bar(x + width/2, l13_means, width, yerr=l13_stds,
                    label='Levels 1-3', color=COLORS['level_1_3'],
                    edgecolor='black', linewidth=1.5, capsize=5)
    
    # Add value labels on bars
    for bar, val in zip(bars1, l12_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, l13_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.axhline(y=0.25, color=COLORS['random_guess'], linestyle='--', linewidth=2,
                label='Random Guessing (25%)')
    ax1.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Mean Balanced Accuracy', fontweight='bold', fontsize=12)
    ax1.set_title('Figure 1: Dataset Comparison - Level 1-2 vs Level 1-3', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 1.0)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'figure1_dataset_comparison.png'), dpi=300, 
                 bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    # =========================================================================
    # Figure 2: Performance Comparison - 4 panels with MultiClass (solid) vs OvR (striped)
    # =========================================================================
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
    metrics = [('Accuracy', 'Mean Balanced Accuracy'), ('Precision', 'Precision'), 
               ('F1', 'F1 Score'), ('Recall', 'Recall')]
    
    for ax, (metric, ylabel) in zip(axes.flatten(), metrics):
        col_mean = f'{metric}_Mean'
        col_std = f'{metric}_Std'
        
        x = np.arange(len(models))
        width = 0.2
        
        # Get data for each combination
        for i, (dataset, dataset_color) in enumerate([('Level 1-2', COLORS['level_1_2']), 
                                                       ('Level 1-3', COLORS['level_1_3'])]):
            for j, strategy in enumerate(['MultiClass', 'OvR']):
                subset = results_df[(results_df['Dataset'] == dataset) & 
                                   (results_df['Strategy'] == strategy)]
                
                means = [subset[subset['Model'] == m][col_mean].values[0] if len(subset[subset['Model'] == m]) > 0 else 0 for m in models]
                stds = [subset[subset['Model'] == m][col_std].values[0] if len(subset[subset['Model'] == m]) > 0 else 0 for m in models]
                
                offset = (i * 2 + j - 1.5) * width
                hatch = '' if strategy == 'MultiClass' else '///'
                
                bars = ax.bar(x + offset, means, width, yerr=stds,
                             color=dataset_color, edgecolor='black', linewidth=1,
                             hatch=hatch, capsize=3, alpha=0.9)
        
        ax.axhline(y=0.25, color=COLORS['random_guess'], linestyle='--', linewidth=2)
        ax.set_xlabel('Model', fontweight='bold', fontsize=11)
        ax.set_ylabel(ylabel, fontweight='bold', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.set_ylim(0, 1.0)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['level_1_2'], edgecolor='black', label='Levels 1-2'),
        Patch(facecolor=COLORS['level_1_3'], edgecolor='black', label='Levels 1-3'),
        Patch(facecolor='white', edgecolor='black', label='Multi-Class (solid)'),
        Patch(facecolor='white', edgecolor='black', hatch='///', label='One-vs-Rest (striped)'),
        plt.Line2D([0], [0], color=COLORS['random_guess'], linestyle='--', linewidth=2, label='Random Guessing (25%)')
    ]
    fig2.legend(handles=legend_elements, loc='upper center', ncol=5, fontsize=10, 
                bbox_to_anchor=(0.5, 1.02))
    
    fig2.suptitle('Figure 2: Model Performance Comparison Across Datasets and Strategies', 
                  fontweight='bold', fontsize=14, y=1.06)
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'figure2_performance_comparison.png'), dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    # =========================================================================
    # Figure 3: Heatmap (unchanged)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(12, 8), facecolor='white')
    
    pivot = results_df.pivot_table(values='Accuracy_Mean', 
                                    index=['Dataset', 'Strategy'], 
                                    columns='Model')
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.4,
                vmin=0.25, vmax=0.65, ax=ax3, linewidths=0.5,
                cbar_kws={'label': 'Balanced Accuracy'})
    ax3.set_title('Figure 3: Model Selection Heatmap', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'figure3_scenario_heatmap.png'), dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    
    # =========================================================================
    # Figure 4: Top 4 Performers - Grouped bar chart with all metrics
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(14, 7), facecolor='white')
    
    top4 = results_df.nlargest(4, 'Accuracy_Mean')
    
    # Create short labels for top 4
    labels = []
    for _, r in top4.iterrows():
        dataset_short = 'Level 1' if 'Level 1-2' in r['Dataset'] else 'Level 1-3'
        strategy_short = 'Multi' if r['Strategy'] == 'MultiClass' else 'OvR'
        model_short = r['Model'][:6] if len(r['Model']) > 6 else r['Model']
        labels.append(f"{dataset_short} | {strategy_short} | {model_short}")
    
    x = np.arange(len(labels))
    width = 0.2
    
    metrics_to_plot = ['Accuracy', 'F1', 'Precision', 'Recall']
    colors_metrics = [COLORS['level_1_2'], COLORS['level_1_3'], 
                      COLORS['multiclass'], COLORS['ovr']]
    
    for i, metric in enumerate(metrics_to_plot):
        col = f'{metric}_Mean'
        values = top4[col].values
        bars = ax4.bar(x + (i - 1.5) * width, values, width, 
                       label=metric, color=colors_metrics[i],
                       edgecolor='black', linewidth=1.5)
    
    ax4.axhline(y=0.25, color=COLORS['random_guess'], linestyle='--', linewidth=2,
                label='Random Guessing (25%)')
    ax4.set_xlabel('Scenario', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax4.set_title('Figure 4: Top 4 Performing Scenarios - Metric Comparison', fontweight='bold', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.legend(loc='upper right', fontsize=10)
    ax4.set_ylim(0, 1.0)
    
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'figure4_top_performers.png'), dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    
    print("Saved: Figures 1-4")

# =============================================================================
# PART 2: HYPERPARAMETER OPTIMIZATION
# =============================================================================

def run_hyperparameter_optimization(sample_details, df_level_1_2, output_dir):
    """
    Intensive hyperparameter optimization on the best model from Part 1.
    Best model: Level 1-2 + MultiClass + LogisticRegression (Elastic Net)
    """
    
    print("=" * 80)
    print("PART 2: HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Running intensive Optuna optimization ({N_TRIALS_OPTIMIZATION} trials)")
    print()
    
    # Prepare data
    le = LabelEncoder()
    merged = df_level_1_2.merge(sample_details, left_on='Sample', right_on='Sample_ID')
    feature_cols = [c for c in df_level_1_2.columns if c != 'Sample']
    X_raw = merged[feature_cols].values.astype(float)
    y = le.fit_transform(merged['Sampling Point'].values)
    
    # Preprocessing
    var_thresh = VarianceThreshold(threshold=0)
    X_var = var_thresh.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var)
    
    n_features = X_scaled.shape[1]
    print(f"Features: {n_features}")
    print()
    
    cv = StratifiedKFold(n_splits=N_FOLDS_OUTER, shuffle=True, random_state=RANDOM_STATE)
    
    # Baseline (5 trials)
    print("Running baseline optimization (5 trials)...")
    baseline_trials = []
    
    def baseline_objective(trial):
        C = trial.suggest_float('C', 1e-4, 100, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        model = LogisticRegression(
            penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio,
            class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE, n_jobs=N_JOBS
        )
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='balanced_accuracy', n_jobs=N_JOBS)
        mean_score = np.mean(scores)
        baseline_trials.append(mean_score)
        return mean_score
    
    study_baseline = optuna.create_study(direction='maximize',
                                          sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_baseline.optimize(baseline_objective, n_trials=5, show_progress_bar=False)
    
    baseline_C = study_baseline.best_params['C']
    baseline_l1 = study_baseline.best_params['l1_ratio']
    
    # Evaluate baseline
    baseline_model = LogisticRegression(
        penalty='elasticnet', solver='saga', C=baseline_C, l1_ratio=baseline_l1,
        class_weight='balanced', max_iter=2000, random_state=RANDOM_STATE, n_jobs=N_JOBS
    )
    
    baseline_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
    for train_idx, val_idx in cv.split(X_scaled, y):
        baseline_model.fit(X_scaled[train_idx], y[train_idx])
        y_pred = baseline_model.predict(X_scaled[val_idx])
        baseline_metrics['accuracy'].append(balanced_accuracy_score(y[val_idx], y_pred))
        baseline_metrics['f1'].append(f1_score(y[val_idx], y_pred, average='weighted', zero_division=0))
        baseline_metrics['precision'].append(precision_score(y[val_idx], y_pred, average='weighted', zero_division=0))
        baseline_metrics['recall'].append(recall_score(y[val_idx], y_pred, average='weighted', zero_division=0))
    
    print(f"  Baseline Accuracy: {np.mean(baseline_metrics['accuracy']):.4f}")
    print()
    
    # Optimized (100 trials)
    print(f"Running intensive optimization ({N_TRIALS_OPTIMIZATION} trials)...")
    optimized_trials = []
    
    def optimized_objective(trial):
        C = trial.suggest_float('C', 1e-5, 1000, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        model = LogisticRegression(
            penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio,
            class_weight='balanced', max_iter=3000, random_state=RANDOM_STATE, n_jobs=N_JOBS
        )
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='balanced_accuracy', n_jobs=N_JOBS)
        mean_score = np.mean(scores)
        optimized_trials.append(mean_score)
        if len(optimized_trials) % 20 == 0:
            print(f"    Trial {len(optimized_trials)}/{N_TRIALS_OPTIMIZATION}: Best = {max(optimized_trials):.4f}")
        return mean_score
    
    study_optimized = optuna.create_study(direction='maximize',
                                           sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE+1))
    study_optimized.optimize(optimized_objective, n_trials=N_TRIALS_OPTIMIZATION, show_progress_bar=False)
    
    optimized_C = study_optimized.best_params['C']
    optimized_l1 = study_optimized.best_params['l1_ratio']
    
    # Evaluate optimized
    optimized_model = LogisticRegression(
        penalty='elasticnet', solver='saga', C=optimized_C, l1_ratio=optimized_l1,
        class_weight='balanced', max_iter=3000, random_state=RANDOM_STATE, n_jobs=N_JOBS
    )
    
    optimized_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
    for train_idx, val_idx in cv.split(X_scaled, y):
        optimized_model.fit(X_scaled[train_idx], y[train_idx])
        y_pred = optimized_model.predict(X_scaled[val_idx])
        optimized_metrics['accuracy'].append(balanced_accuracy_score(y[val_idx], y_pred))
        optimized_metrics['f1'].append(f1_score(y[val_idx], y_pred, average='weighted', zero_division=0))
        optimized_metrics['precision'].append(precision_score(y[val_idx], y_pred, average='weighted', zero_division=0))
        optimized_metrics['recall'].append(recall_score(y[val_idx], y_pred, average='weighted', zero_division=0))
    
    print()
    print(f"  Optimized Accuracy: {np.mean(optimized_metrics['accuracy']):.4f}")
    
    improvement = (np.mean(optimized_metrics['accuracy']) - np.mean(baseline_metrics['accuracy'])) / np.mean(baseline_metrics['accuracy']) * 100
    print(f"  Improvement: {improvement:+.1f}%")
    print()
    
    # Save results
    summary_df = pd.DataFrame({
        'Model': ['Baseline (5 trials)', f'Optimized ({N_TRIALS_OPTIMIZATION} trials)'],
        'N_Features': [n_features, n_features],
        'Accuracy_Mean': [np.mean(baseline_metrics['accuracy']), np.mean(optimized_metrics['accuracy'])],
        'Accuracy_Std': [np.std(baseline_metrics['accuracy']), np.std(optimized_metrics['accuracy'])],
        'F1_Mean': [np.mean(baseline_metrics['f1']), np.mean(optimized_metrics['f1'])],
        'Precision_Mean': [np.mean(baseline_metrics['precision']), np.mean(optimized_metrics['precision'])],
        'Recall_Mean': [np.mean(baseline_metrics['recall']), np.mean(optimized_metrics['recall'])],
        'C': [baseline_C, optimized_C],
        'l1_ratio': [baseline_l1, optimized_l1]
    })
    summary_df.to_csv(os.path.join(output_dir, 'optimization_summary.csv'), index=False)
    
    # Generate figures
    generate_optimization_figures(baseline_metrics, optimized_metrics, optimized_trials, 
                                  n_features, output_dir)
    
    return {
        'optimized_C': optimized_C,
        'optimized_l1_ratio': optimized_l1,
        'baseline_metrics': baseline_metrics,
        'optimized_metrics': optimized_metrics,
        'optimized_trials': optimized_trials
    }

def generate_optimization_figures(baseline_metrics, optimized_metrics, trial_values, n_features, output_dir):
    """Generate hyperparameter optimization visualization figures."""
    
    # Figure 5: Optimization Comparison
    fig5, ax5 = plt.subplots(figsize=(10, 7), facecolor='white')
    
    metrics_labels = ['Balanced\nAccuracy', 'F1 Score', 'Precision', 'Recall']
    baseline_values = [np.mean(baseline_metrics['accuracy']), np.mean(baseline_metrics['f1']),
                       np.mean(baseline_metrics['precision']), np.mean(baseline_metrics['recall'])]
    baseline_errors = [np.std(baseline_metrics['accuracy']), np.std(baseline_metrics['f1']),
                       np.std(baseline_metrics['precision']), np.std(baseline_metrics['recall'])]
    optimized_values = [np.mean(optimized_metrics['accuracy']), np.mean(optimized_metrics['f1']),
                        np.mean(optimized_metrics['precision']), np.mean(optimized_metrics['recall'])]
    optimized_errors = [np.std(optimized_metrics['accuracy']), np.std(optimized_metrics['f1']),
                        np.std(optimized_metrics['precision']), np.std(optimized_metrics['recall'])]
    
    x = np.arange(len(metrics_labels))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, baseline_values, width, yerr=baseline_errors,
                    label='Baseline (5 trials)', color=COLORS['baseline'], 
                    edgecolor='black', linewidth=1.5, capsize=5)
    bars2 = ax5.bar(x + width/2, optimized_values, width, yerr=optimized_errors,
                    label='Optimized (100 trials)', color=COLORS['optimized'],
                    edgecolor='black', linewidth=1.5, capsize=5)
    
    ax5.axhline(y=0.25, color=COLORS['random_guess'], linestyle='--', linewidth=2,
                label='Random Guessing (25%)')
    
    ax5.set_xlabel('Metric', fontweight='bold')
    ax5.set_ylabel('Score', fontweight='bold')
    ax5.set_title(f'Figure 5: Hyperparameter Optimization Results\n({n_features} features)',
                  fontweight='bold', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics_labels)
    ax5.legend(loc='upper right')
    ax5.set_ylim(0, 1.0)
    
    for bar, val in zip(bars1, baseline_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, optimized_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    fig5.savefig(os.path.join(output_dir, 'figure5_optimization_comparison.png'), dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig5)
    
    # Figure 6: Optimization Curve
    fig6, ax6 = plt.subplots(figsize=(12, 7), facecolor='white')
    
    trials = range(1, len(trial_values) + 1)
    running_best = np.maximum.accumulate(trial_values)
    
    ax6.scatter(trials, trial_values, color=COLORS['multiclass'], alpha=0.5, s=30,
                label='Individual Trial Scores')
    ax6.plot(trials, running_best, color=COLORS['optimized'], linewidth=2.5,
             label='Running Best Accuracy')
    
    best_idx = np.argmax(trial_values)
    ax6.scatter([best_idx + 1], [trial_values[best_idx]], color=COLORS['highlight'], s=200,
                zorder=5, edgecolor='black', linewidth=2, marker='*', label=f'Best: Trial #{best_idx+1}')
    
    ax6.axhline(y=np.mean(baseline_values), color=COLORS['baseline'], linestyle='--', linewidth=2,
                label=f'Baseline ({np.mean(baseline_values):.3f})')
    ax6.axhline(y=0.25, color=COLORS['random_guess'], linestyle='--', linewidth=2,
                label='Random Guessing (25%)')
    
    ax6.set_xlabel('Optuna Trial Number', fontweight='bold')
    ax6.set_ylabel('Cross-Validated Balanced Accuracy', fontweight='bold')
    ax6.set_title('Figure 6: Hyperparameter Optimization Progress', fontweight='bold', fontsize=14)
    ax6.legend(loc='lower right')
    
    plt.tight_layout()
    fig6.savefig(os.path.join(output_dir, 'figure6_optuna_optimization_curve.png'), dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig6)
    
    print("Saved: Figures 5-6")

# =============================================================================
# PART 3: VALIDATION AND INTERPRETATION
# =============================================================================

def run_validation(sample_details, df_level_1_2, optimized_params, output_dir, model_selection_results=None):
    """
    Final validation with cross_val_predict and SHAP analysis.
    """
    
    print("=" * 80)
    print("PART 3: VALIDATION AND INTERPRETATION")
    print("=" * 80)
    print()
    
    # Prepare data
    le = LabelEncoder()
    merged = df_level_1_2.merge(sample_details, left_on='Sample', right_on='Sample_ID')
    feature_cols = [c for c in df_level_1_2.columns if c != 'Sample']
    X_raw = merged[feature_cols].values.astype(float)
    y = le.fit_transform(merged['Sampling Point'].values)
    class_names = le.classes_
    sample_ids = merged['Sample'].values
    
    # Preprocessing
    var_thresh = VarianceThreshold(threshold=0)
    X_var = var_thresh.fit_transform(X_raw)
    feature_mask = var_thresh.get_support()
    feature_names = [feature_cols[i] for i in range(len(feature_cols)) if feature_mask[i]]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var)
    
    n_features = X_scaled.shape[1]
    n_classes = len(class_names)
    
    print(f"Features: {n_features}")
    print(f"Samples: {X_scaled.shape[0]}")
    print(f"Classes: {n_classes}")
    print()
    
    cv = StratifiedKFold(n_splits=N_FOLDS_OUTER, shuffle=True, random_state=RANDOM_STATE)
    
    # Create optimized model
    model = LogisticRegression(
        penalty='elasticnet', solver='saga',
        C=optimized_params['optimized_C'], l1_ratio=optimized_params['optimized_l1_ratio'],
        class_weight='balanced', max_iter=3000, random_state=RANDOM_STATE, n_jobs=N_JOBS
    )
    
    # Out-of-fold predictions
    print("Generating out-of-fold predictions...")
    y_pred = cross_val_predict(model, X_scaled, y, cv=cv, method='predict', n_jobs=N_JOBS)
    y_pred_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba', n_jobs=N_JOBS)
    
    # Calculate metrics
    accuracy = balanced_accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    print(f"\nFINAL CROSS-VALIDATION METRICS:")
    print(f"  Balanced Accuracy: {accuracy:.4f}")
    print(f"  F1 Score:          {f1:.4f}")
    print(f"  Precision:         {precision:.4f}")
    print(f"  Recall:            {recall:.4f}")
    print()
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Sample_ID': sample_ids,
        'True_Label': [class_names[i] for i in y],
        'Predicted_Label': [class_names[i] for i in y_pred],
        'Correct': y == y_pred
    })
    for i in range(n_classes):
        predictions_df[f'Prob_Class_{i}'] = y_pred_proba[:, i]
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # SHAP Analysis
    print("Calculating SHAP values...")
    model.fit(X_scaled, y)
    explainer = shap.LinearExplainer(model, X_scaled, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_scaled)
    
    # Calculate mean SHAP importance
    shap_array = np.array(shap_values)
    if len(shap_array.shape) == 3:
        mean_shap = np.mean(np.abs(shap_array), axis=(0, 2))
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)
    
    sorted_indices = np.argsort(mean_shap)[::-1]
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'Rank': range(1, len(feature_names) + 1),
        'Feature': [feature_names[int(i)] for i in sorted_indices],
        'Mean_Abs_SHAP': [mean_shap[int(i)] for i in sorted_indices]
    })
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # Find representative samples
    representative_samples = {}
    for class_idx, class_name in enumerate(class_names):
        correct_mask = (y == class_idx) & (y_pred == class_idx)
        correct_indices = np.where(correct_mask)[0]
        if len(correct_indices) > 0:
            probs = y_pred_proba[correct_indices, class_idx]
            best_idx = correct_indices[np.argmax(probs)]
            representative_samples[class_idx] = {
                'index': best_idx,
                'sample_id': sample_ids[best_idx],
                'probability': y_pred_proba[best_idx, class_idx],
                'class_name': class_name
            }
    
    # Generate figures
    generate_validation_figures(X_scaled, y, y_pred, cm, class_names, feature_names,
                                shap_values, mean_shap, explainer, representative_samples,
                                accuracy, f1, precision, recall, output_dir,
                                model_selection_results, optimized_params)
    
    # Save final metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Balanced_Accuracy', 'F1_Score', 'Precision', 'Recall'],
        'Value': [accuracy, f1, precision, recall]
    })
    metrics_df.to_csv(os.path.join(output_dir, 'final_metrics.csv'), index=False)
    
    print()
    print("PART 3 COMPLETE")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm
    }

def generate_validation_figures(X_scaled, y, y_pred, cm, class_names, feature_names,
                                 shap_values, mean_shap, explainer, representative_samples,
                                 accuracy, f1, precision, recall, output_dir,
                                 model_selection_results=None, optimized_params=None):
    """Generate validation and SHAP visualization figures."""
    
    n_classes = len(class_names)
    shap_array = np.array(shap_values)
    
    # Figure 7: SHAP Summary
    print("Generating Figure 7: SHAP Summary...")
    fig7, ax7 = plt.subplots(figsize=(12, 10), facecolor='white')
    
    if len(shap_array.shape) == 3:
        shap_for_plot = np.mean(np.abs(shap_array), axis=2)
    else:
        shap_for_plot = shap_array
    
    plt.sca(ax7)
    shap.summary_plot(shap_for_plot, X_scaled, feature_names=feature_names,
                      max_display=20, show=False, plot_size=None)
    ax7.set_title('Figure 7: SHAP Feature Importance Summary\n(Top 20 Features)',
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    fig7.savefig(os.path.join(output_dir, 'figure7_shap_summary.png'), dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig7)
    
    # Figure 8a-8d: SHAP Waterfall Plots (separate figures)
    print("Generating Figure 8a-8d: SHAP Waterfall Plots...")
    
    for class_idx in range(n_classes):
        if class_idx in representative_samples:
            sample_info = representative_samples[class_idx]
            sample_idx = sample_info['index']
            
            if len(shap_array.shape) == 3:
                sv = shap_array[sample_idx, :, class_idx]
                base_val = explainer.expected_value[class_idx] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
            else:
                sv = shap_values[sample_idx]
                base_val = explainer.expected_value
            
            fig8, ax8 = plt.subplots(figsize=(10, 8), facecolor='white')
            
            explanation = shap.Explanation(
                values=sv, base_values=base_val,
                data=X_scaled[sample_idx], feature_names=feature_names
            )
            
            plt.sca(ax8)
            shap.plots.waterfall(explanation, max_display=10, show=False)
            
            location_name = sample_info['class_name']
            if ' - ' in location_name:
                location_name = location_name.split(' - ', 1)[1]
            
            ax8.set_title(f"Figure 8{chr(97+class_idx)}: SHAP Waterfall - {location_name}",
                         fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            safe_name = location_name.replace(" ", "_").lower()
            fig8.savefig(os.path.join(output_dir, f'figure8{chr(97+class_idx)}_shap_waterfall_{safe_name}.png'),
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig8)
    
    # Figure 9: 4-Panel Summary (matching desired layout)
    print("Generating Figure 9: Summary Figure...")
    fig9 = plt.figure(figsize=(16, 14), facecolor='white')
    gs = GridSpec(2, 2, figure=fig9, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel A: Model Selection Results (grouped bar chart by model)
    # =========================================================================
    ax_a = fig9.add_subplot(gs[0, 0])
    
    if model_selection_results is not None:
        models = ['LinearSVC', 'LogisticRegression', 'RandomForest', 'XGBoost']
        model_labels = ['Linear\nSVC', 'Logistic\nRegression', 'Random\nForest', 'XGBoost']
        x = np.arange(len(models))
        width = 0.35
        
        l12_means = []
        l13_means = []
        for model in models:
            l12_data = model_selection_results[(model_selection_results['Dataset'] == 'Level 1-2') & 
                                                (model_selection_results['Model'] == model)]
            l13_data = model_selection_results[(model_selection_results['Dataset'] == 'Level 1-3') & 
                                                (model_selection_results['Model'] == model)]
            l12_means.append(l12_data['Accuracy_Mean'].mean() if len(l12_data) > 0 else 0)
            l13_means.append(l13_data['Accuracy_Mean'].mean() if len(l13_data) > 0 else 0)
        
        bars1 = ax_a.bar(x - width/2, l12_means, width, label='Level 1-2', 
                         color=COLORS['level_1_2'], edgecolor='black', linewidth=1.5)
        bars2 = ax_a.bar(x + width/2, l13_means, width, label='Level 1-3',
                         color=COLORS['level_1_3'], edgecolor='black', linewidth=1.5)
        
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(model_labels, fontsize=10)
    
    ax_a.axhline(y=0.25, color=COLORS['random_guess'], linestyle='--', linewidth=2, label='Random (25%)')
    ax_a.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax_a.set_ylabel('Balanced Accuracy', fontweight='bold', fontsize=11)
    ax_a.set_title('A) Phase 1: Model Selection Results', fontweight='bold', fontsize=12)
    ax_a.legend(loc='upper right', fontsize=9)
    ax_a.set_ylim(0, 0.8)
    
    # =========================================================================
    # Panel B: Hyperparameter Optimization Curve
    # =========================================================================
    ax_b = fig9.add_subplot(gs[0, 1])
    
    if optimized_params is not None and 'optimized_trials' in optimized_params:
        trial_values = optimized_params['optimized_trials']
        trials = range(1, len(trial_values) + 1)
        running_best = np.maximum.accumulate(trial_values)
        
        ax_b.scatter(trials, trial_values, color='lightblue', alpha=0.6, s=20, label='Trial Scores')
        ax_b.plot(trials, running_best, color=COLORS['level_1_3'], linewidth=2.5, label='Running Best')
        
        best_idx = np.argmax(trial_values)
        ax_b.scatter([best_idx + 1], [trial_values[best_idx]], color=COLORS['highlight'], s=200,
                    zorder=5, edgecolor='black', linewidth=2, marker='*', 
                    label=f'Best: {trial_values[best_idx]:.1%}')
        
        baseline_acc = np.mean(optimized_params['baseline_metrics']['accuracy'])
        ax_b.axhline(y=baseline_acc, color=COLORS['level_1_2'], linestyle='--', linewidth=2,
                    label=f'Baseline ({baseline_acc:.1%})')
    
    ax_b.axhline(y=0.25, color=COLORS['random_guess'], linestyle='--', linewidth=2, label='Random (25%)')
    ax_b.set_xlabel('Optuna Trial', fontweight='bold', fontsize=11)
    ax_b.set_ylabel('CV Balanced Accuracy', fontweight='bold', fontsize=11)
    ax_b.set_title('B) Phase 2.2: Hyperparameter Optimization', fontweight='bold', fontsize=12)
    ax_b.legend(loc='lower right', fontsize=8)
    ax_b.set_ylim(0.2, 0.7)
    
    # =========================================================================
    # Panel C: Final Cross-Validation Metrics
    # =========================================================================
    ax_c = fig9.add_subplot(gs[1, 0])
    metrics_names = ['Balanced\nAccuracy', 'F1 Score', 'Precision', 'Recall']
    values = [accuracy, f1, precision, recall]
    colors_metrics = [COLORS['level_1_2'], COLORS['level_1_3'], COLORS['multiclass'], COLORS['ovr']]
    
    bars_c = ax_c.bar(metrics_names, values, color=colors_metrics, edgecolor='black', linewidth=1.5)
    ax_c.axhline(y=0.25, color=COLORS['random_guess'], linestyle='--', linewidth=2, label='Random (25%)')
    
    for bar, val in zip(bars_c, values):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax_c.set_ylabel('Score', fontweight='bold', fontsize=11)
    ax_c.set_title('C) Phase 3: Final Cross-Validation Metrics', fontweight='bold', fontsize=12)
    ax_c.set_ylim(0, 1.0)
    ax_c.legend(loc='upper right', fontsize=9)
    
    # =========================================================================
    # Panel D: Confusion Matrix
    # =========================================================================
    ax_d = fig9.add_subplot(gs[1, 1])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax_d.imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', fontweight='bold')
    
    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm_normalized[i, j] > 0.5 else 'black'
            ax_d.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.0%})',
                     ha='center', va='center', color=color, fontsize=10, fontweight='bold')
    
    # Create short location names
    short_names = []
    for name in class_names:
        if 'Agias' in name or 'agias' in name.lower():
            short_names.append('Agias Sofias')
        elif 'Ethnikis' in name or 'ethnikis' in name.lower():
            short_names.append('Ethnikis Ami')
        elif 'Lagkada' in name or 'lagkada' in name.lower():
            short_names.append('Lagkada Str')
        elif 'Port' in name or 'port' in name.lower() or 'Thessaloniki' in name:
            short_names.append('Thessaloniki')
        else:
            short_names.append(name[:12])
    
    ax_d.set_xticks(range(n_classes))
    ax_d.set_yticks(range(n_classes))
    ax_d.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax_d.set_yticklabels(short_names, fontsize=9)
    ax_d.set_xlabel('Predicted Label', fontweight='bold', fontsize=11)
    ax_d.set_ylabel('True Label', fontweight='bold', fontsize=11)
    ax_d.set_title('D) Confusion Matrix (Cross-Validation)', fontweight='bold', fontsize=12)
    
    fig9.suptitle('Figure 9: NTA Stormwater Classification - Complete Analysis Summary', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig9.savefig(os.path.join(output_dir, 'figure9_summary_4panel.png'), dpi=300,
                 bbox_inches='tight', facecolor='white')
    plt.close(fig9)
    
    print("Saved: Figures 7-9")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("NTA STORMWATER CLASSIFICATION PIPELINE")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set file paths (files should be in the same directory as this script)
    data_dir = '.'
    output_dir = OUTPUT_DIR
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    sample_details = load_sample_details(os.path.join(data_dir, 'sample_details.txt'))
    df_level_1_2 = load_identifications(os.path.join(data_dir, 'CL_1_2a_2b_identifications.txt'))
    df_level_1_3 = load_identifications(os.path.join(data_dir, 'CL_1_2a_2b_3_identifications.txt'))
    
    print(f"  Level 1-2: {len(df_level_1_2.columns) - 1} features")
    print(f"  Level 1-3: {len(df_level_1_3.columns) - 1} features")
    print(f"  Samples: {len(sample_details)}")
    print()
    
    # Part 1: Model Selection
    results_df, best_model = run_model_selection(sample_details, df_level_1_2, df_level_1_3, output_dir)
    
    # Part 2: Hyperparameter Optimization
    optimized_params = run_hyperparameter_optimization(sample_details, df_level_1_2, output_dir)
    
    # Part 3: Validation and Interpretation
    final_results = run_validation(sample_details, df_level_1_2, optimized_params, output_dir, results_df)
    
    print()
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Final Accuracy: {final_results['accuracy']:.1%}")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
