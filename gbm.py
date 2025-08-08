#!/usr/bin/env python3
"""Gradient Boosting Machine training script for spatial and temporal data modeling.

This module implements a dual-model approach (spatial and temporal) using LightGBM
with attention-inspired features and ultra-high capacity models designed for
handling extreme variations (350x+) without hardcoded geometric assumptions.

The approach is based on universal feature learning principles:
- No predefined spatial thresholds or region boundaries
- Attention-inspired feature selection mechanisms
- Ultra-high model capacity to naturally learn extreme differences
- Data-driven rather than assumption-driven modeling

Example usage:
    python gbm.py -i spatial_data.parquet temporal_data.parquet -o model.pkl
"""

import argparse
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import pickle
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split

from config import *

# Universal multi-scale parameters (no geometric assumptions)
MULTI_SCALE_RADIAL = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]  # Multiple scales for learning
MULTI_SCALE_ANGULAR = [1, 2, 3, 4, 6, 8, 12, 16]           # Multiple angular frequencies
MULTI_SCALE_TEMPORAL = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]  # Multiple time scales
GBM_RADIAL_KNOTS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

# JUNO detector geometry constants (physics-based, universal)
R0_PHYSICAL = 17.7   # Liquid scintillator sphere radius (m)
R_PMT_PHYSICAL = 19.5  # PMT sphere radius (m)
LIGHT_SPEED_IN_LS = 2.998e8 / 1.478  # m/ns in liquid scintillator (n=1.478)


def calculate_universal_physics_features(r_norm, theta):
    """Calculate universal physics-based features without geometric assumptions.

    These features are based on fundamental physical laws that apply regardless
    of detector configuration or specific geometry.

    Args:
        r_norm (np.ndarray): Normalized radial distance [0,1].
        theta (np.ndarray): Angular coordinate in radians.

    Returns:
        list: List of universal physics-based features.
    """
    # Convert to physical coordinates (universal conversion)
    r_physical = r_norm * R0_PHYSICAL

    # True distance calculation using cosine rule (universal geometric law)
    cos_theta = np.cos(theta)
    d_squared = r_physical**2 + R_PMT_PHYSICAL**2 - 2*r_physical*R_PMT_PHYSICAL*cos_theta
    d_real = np.sqrt(d_squared + 1e-12)  # Small epsilon to avoid division by zero

    # Universal physics-based features
    physics_features = []

    # 1. Inverse square law intensity (universal for light propagation)
    intensity = 1.0 / (d_real**2 + 1.0)
    physics_features.append(intensity)

    # 2. Multiple scale distance features (no assumption about optimal scale)
    for scale in [0.5, 1.0, 2.0, 4.0]:
        scaled_distance = d_real / scale
        physics_features.extend([
            1 / (1 + scaled_distance),           # Multiple scale inverse
            np.exp(-scaled_distance),            # Multiple scale exponential
            1 / (1 + scaled_distance**2),        # Multiple scale inverse quadratic
        ])

    # 3. Light path geometry features (universal geometric relationships)
    # Angle between radial vector and line to PMT
    path_angle = np.arccos(np.clip(cos_theta, -1, 1))
    physics_features.extend([
        np.sin(path_angle),                      # Path geometry sine
        np.cos(path_angle),                      # Path geometry cosine
        path_angle / np.pi,                      # Normalized path angle
        np.sin(2 * path_angle),                  # Double angle components
        np.cos(2 * path_angle),
    ])

    return physics_features


def create_attention_inspired_spatial_features(r_norm, theta):
    """Create attention-inspired spatial features that let model learn focus regions.

    Instead of hardcoding specific regions (like r>0.8, θ<π/6), this function
    creates multiple "attention heads" that the model can learn to use selectively.

    Args:
        r_norm (np.ndarray): Normalized radial distance [0,1].
        theta (np.ndarray): Angular coordinate in radians.

    Returns:
        list: List of attention-inspired spatial features.
    """
    attention_features = []

    # Multi-center radial attention (let model learn important radial regions)
    for i, center_r in enumerate(np.linspace(0.0, 1.2, 10)):  # 10 potential focus centers
        # Gaussian attention weight centered at center_r
        for width in [0.1, 0.3, 0.6]:  # Multiple attention widths
            radial_attention = np.exp(-((r_norm - center_r)**2) / (2 * width**2))
            attention_features.append(radial_attention)

    # Multi-directional angular attention (let model learn important angular regions)
    for i, center_theta in enumerate(np.linspace(0, 2*np.pi, 12)):  # 12 potential directions
        # Handle circular nature of angles
        theta_diff = np.abs(theta - center_theta)
        theta_diff = np.minimum(theta_diff, 2*np.pi - theta_diff)  # Circular distance

        for width in [np.pi/8, np.pi/4, np.pi/2]:  # Multiple angular attention widths
            angular_attention = np.exp(-(theta_diff**2) / (2 * width**2))
            attention_features.append(angular_attention)
    focused_centers_deg = [-15, -10, -5, 0, 5, 10, 15]
    focused_centers_rad = np.deg2rad(focused_centers_deg)

    for center_theta in focused_centers_rad:
        # Compute angular difference to each center (handle circular wrap-around)
        theta_diff = np.abs(theta - center_theta)
        theta_diff = np.minimum(theta_diff, 2*np.pi - theta_diff)

        # Use two different 'spotlight' widths: narrow and wide
        # Narrow spotlight for precise focus
        narrow_attention = np.exp(-(theta_diff**2) / (2 * (np.pi/18)**2))  # 10° width
        # Wide spotlight covering a broader region
        wide_attention = np.exp(-(theta_diff**2) / (2 * (np.pi/9)**2))     # 20° width

        attention_features.append(narrow_attention)
        attention_features.append(wide_attention)

    # Combined radial-angular attention patches (2D attention)
    for r_center in [0.2, 0.5, 0.8, 0.9, 1.0]:
        for theta_center in np.linspace(0, 2*np.pi, 8):
            # 2D Gaussian attention patch
            theta_diff = np.abs(theta - theta_center)
            theta_diff = np.minimum(theta_diff, 2*np.pi - theta_diff)

            combined_attention = np.exp(-(
                (r_norm - r_center)**2 / (2 * 0.3**2) +
                theta_diff**2 / (2 * (np.pi/4)**2)
            ))
            attention_features.append(combined_attention)

    return attention_features


def create_multi_scale_universal_features(x, y):
    """Create multi-scale universal features without geometric assumptions.

    This function creates features at multiple scales and patterns, allowing
    the model to automatically learn which scales and patterns are important
    for different regions without hardcoding assumptions.

    Args:
        x (np.ndarray): X coordinates.
        y (np.ndarray): Y coordinates.

    Returns:
        list: List of multi-scale universal features.
    """
    # Normalize coordinates
    x_norm = x / R0
    y_norm = y / R0
    r_norm = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)

    multi_scale_features = []

    # Multi-scale radial features (no assumption about optimal scale)
    for scale in MULTI_SCALE_RADIAL:
        r_scaled = r_norm / scale
        multi_scale_features.extend([
            np.exp(-r_scaled),                   # Exponential decay at this scale
            np.exp(-r_scaled**2),                # Gaussian decay at this scale
            1 / (1 + r_scaled),                  # Inverse decay at this scale
            1 / (1 + r_scaled**2),               # Inverse quadratic at this scale
            r_scaled,                            # Linear at this scale
            r_scaled**2,                         # Quadratic at this scale
        ])

    # Multi-frequency angular features (no assumption about symmetries)
    for freq in MULTI_SCALE_ANGULAR:
        multi_scale_features.extend([
            np.cos(freq * theta),                # Cosine harmonics
            np.sin(freq * theta),                # Sine harmonics
        ])

    # Multi-scale combined radial-angular features
    for r_scale in [0.5, 1.0, 2.0]:
        for ang_freq in [2, 4, 6]:
            r_scaled = r_norm / r_scale
            multi_scale_features.extend([
                r_scaled * np.cos(ang_freq * theta),  # Radial-angular coupling
                r_scaled * np.sin(ang_freq * theta),
                np.exp(-r_scaled) * np.cos(ang_freq * theta),  # Decaying angular patterns
                np.exp(-r_scaled) * np.sin(ang_freq * theta),
            ])

    return multi_scale_features


def create_spatial_features(x, y):
    """Create spatial features using B-spline basis functions and physical constraints.

    Args:
        x (np.ndarray): X coordinates.
        y (np.ndarray): Y coordinates.

    Returns:
        np.ndarray: Feature matrix with spatial basis functions and interactions.
    """
    # Normalize coordinates to [-1, 1] range
    x_norm = x / R0
    y_norm = y / R0
    r_norm = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)

    # B-spline style radial basis functions
    radial_bases = []
    for knot in GBM_RADIAL_KNOTS:
        basis = np.exp(-((r_norm - knot)**2) / (2 * 0.15**2))
        radial_bases.append(basis)

    # Angular basis functions (Fourier basis)
    angular_bases = []
    for k in range(1, 5):
        angular_bases.extend([np.cos(k * theta), np.sin(k * theta)])

    # Radial-angular tensor product interactions
    radial_angular_interactions = []
    for r_basis in radial_bases[:4]:
        for ang_basis in angular_bases[:4]:
            radial_angular_interactions.append(r_basis * ang_basis)

    # Physical constraint features
    r_inv = 1.0 / (r_norm + 0.1)  # Inverse radial distance
    r_inv_squared = r_inv**2
    directional_weight = np.abs(np.cos(theta))
    directional_inv_square = r_inv_squared * directional_weight

    # Total reflection suppression at specific angles
    total_reflection_mask = 1.0 - 0.5 * (
        np.exp(-((theta - np.pi/4)**2) / (2 * (np.pi/6)**2)) +
        np.exp(-((theta - 3*np.pi/4)**2) / (2 * (np.pi/6)**2)) +
        np.exp(-((theta + np.pi/4)**2) / (2 * (np.pi/6)**2)) +
        np.exp(-((theta + 3*np.pi/4)**2) / (2 * (np.pi/6)**2))
    )

    # Combine all features
    feature_list = [
        x_norm, y_norm, r_norm,
        np.cos(theta), np.sin(theta),
        *radial_bases,
        *angular_bases[:6],
        *radial_angular_interactions[:8],
        r_inv, r_inv_squared,
        directional_inv_square,
        total_reflection_mask
    ]

    features = np.column_stack(feature_list)
    print(f"Spatial features: {features.shape[1]} dimensions")

    return features


def create_attention_inspired_temporal_features(r_norm, theta, t_norm):
    """Create attention-inspired temporal features for adaptive pattern learning.

    Instead of predefined temporal templates, this creates multiple temporal
    "attention heads" that the model can learn to combine adaptively.

    Args:
        r_norm (np.ndarray): Normalized radial distance [0,1].
        theta (np.ndarray): Angular coordinate in radians.
        t_norm (np.ndarray): Normalized time coordinate [0,1].

    Returns:
        list: List of attention-inspired temporal features.
    """
    temporal_attention_features = []

    # Multi-center temporal attention (let model learn important time regions)
    for time_center in np.linspace(0.0, 1.2, 15):  # 15 potential temporal focus points
        for width in [0.05, 0.1, 0.2, 0.4]:  # Multiple temporal attention widths
            temporal_attention = np.exp(-((t_norm - time_center)**2) / (2 * width**2))
            temporal_attention_features.append(temporal_attention)

    # Multi-scale temporal decay patterns (no assumption about decay rates)
    for decay_rate in MULTI_SCALE_TEMPORAL:
        temporal_attention_features.extend([
            np.exp(-t_norm * decay_rate),                    # Exponential decay
            t_norm * np.exp(-t_norm * decay_rate),           # Delayed exponential
            (1 - np.exp(-t_norm * decay_rate)),              # Rising exponential
            np.exp(-t_norm * decay_rate) * np.cos(t_norm * decay_rate * 2),  # Oscillating decay
        ])

    # Spatial-conditional temporal attention (let model learn spatial-temporal coupling)
    for r_condition in [0.3, 0.6, 0.9]:  # Different radial conditions
        r_weight = np.exp(-((r_norm - r_condition)**2) / (2 * 0.2**2))
        for t_pattern in [0.1, 0.3, 0.6, 0.9]:  # Different temporal patterns
            t_weight = np.exp(-((t_norm - t_pattern)**2) / (2 * 0.15**2))
            conditional_attention = r_weight * t_weight
            temporal_attention_features.append(conditional_attention)

    # Angular-conditional temporal attention
    for angle_condition in np.linspace(0, np.pi, 6):  # Different angular conditions
        angle_diff = np.abs(theta - angle_condition)
        angle_weight = np.exp(-(angle_diff**2) / (2 * (np.pi/4)**2))
        for t_pattern in [0.1, 0.4, 0.7]:  # Different temporal patterns
            t_weight = np.exp(-((t_norm - t_pattern)**2) / (2 * 0.2**2))
            conditional_attention = angle_weight * t_weight
            temporal_attention_features.append(conditional_attention)

    return temporal_attention_features


def create_temporal_features(x, y, t):
    """Create ULTRA-HIGH-CAPACITY temporal features with attention-inspired design.

    This function creates comprehensive temporal features using:
    1. Multi-scale universal temporal patterns
    2. Attention-inspired adaptive temporal learning
    3. Spatial-temporal interaction features
    4. Physics-based universal temporal features

    The goal is to give the model enough capacity to naturally learn extreme
    temporal variations (350x+) without hardcoded assumptions.

    Args:
        x (np.ndarray): X coordinates.
        y (np.ndarray): Y coordinates.
        t (np.ndarray): Time coordinates.

    Returns:
        np.ndarray: Feature matrix with ultra-comprehensive temporal features.
    """
    # Normalize all coordinates
    x_norm = x / R0
    y_norm = y / R0
    t_norm = t / T_MAX
    r_norm = np.sqrt(x_norm**2 + y_norm**2)
    theta = np.arctan2(y_norm, x_norm)

    # Start with basic universal coordinates
    feature_list = [
        x_norm, y_norm, r_norm, theta, t_norm,  # Basic 5D coordinates
    ]

    # Add universal physics-based features (same as spatial)
    physics_features = calculate_universal_physics_features(r_norm, theta)
    feature_list.extend(physics_features)

    # Add attention-inspired spatial features for spatial context
    spatial_attention_features = create_attention_inspired_spatial_features(r_norm, theta)
    feature_list.extend(spatial_attention_features)

    # Add attention-inspired temporal features (main innovation)
    temporal_attention_features = create_attention_inspired_temporal_features(r_norm, theta, t_norm)
    feature_list.extend(temporal_attention_features)

    # Add multi-scale spatial features for context
    multi_scale_spatial = create_multi_scale_universal_features(x, y)
    feature_list.extend(multi_scale_spatial)

    # Add comprehensive spatial-temporal interaction features
    interaction_features = []

    # Basic interactions
    interaction_features.extend([
        r_norm * t_norm,                     # Radial-temporal coupling
        theta * t_norm,                      # Angular-temporal coupling
        r_norm * theta * t_norm,             # Full 3D interaction
    ])

    # Multi-scale spatial-temporal interactions
    for r_scale in [0.5, 1.0, 2.0]:
        for t_scale in [0.2, 0.5, 1.0, 2.0]:
            r_scaled = r_norm / r_scale
            t_scaled = t_norm / t_scale
            interaction_features.extend([
                r_scaled * np.exp(-t_scaled),           # Radial decay with time
                np.exp(-r_scaled) * t_scaled,           # Temporal growth with radius decay
                r_scaled * t_scaled * np.cos(theta),    # Angular modulated interaction
                r_scaled * t_scaled * np.sin(theta),
            ])

    # Multi-frequency temporal modulations
    for freq in [1, 2, 4, 8]:
        for phase in [0, np.pi/4, np.pi/2]:
            interaction_features.extend([
                r_norm * np.cos(freq * t_norm * 2 * np.pi + phase),  # Temporal oscillations
                theta * np.sin(freq * t_norm * 2 * np.pi + phase),   # Angular temporal oscillations
            ])

    feature_list.extend(interaction_features)

    # Combine all features
    features = np.column_stack(feature_list)

    return features


def train_spatial_model(spatial_data_path):
    """Train spatial model with ultra-high capacity and universal features.

    Args:
        spatial_data_path (str): Path to spatial data parquet file.

    Returns:
        dict: Dictionary containing trained model and preprocessing artifacts.
    """
    print("Training ULTRA-HIGH-CAPACITY spatial model...")

    # Load spatial data
    table_s = pq.read_table(spatial_data_path)
    x_s = table_s["x"].to_numpy()
    y_s = table_s["y"].to_numpy()
    nev_s = table_s["nEV"].to_numpy()  # Number of events
    npe_s = table_s["nPE"].to_numpy()  # Number of photoelectrons

    print(f"Spatial data: {len(x_s)} points")
    print(f"nEV range: [{nev_s.min():.1f}, {nev_s.max():.1f}]")
    print(f"nPE range: [{npe_s.min():.1f}, {npe_s.max():.1f}]")

    # Model response ratio with log transform
    epsilon = 1e-10  # Very small constant to avoid division by zero
    response_ratio_s = (npe_s + epsilon) / (nev_s + epsilon)
    log_response_ratio_s = np.log(response_ratio_s)

    print(f"Response ratio range: [{response_ratio_s.min():.6f}, {response_ratio_s.max():.6f}]")
    print(f"Log response ratio range: [{log_response_ratio_s.min():.3f}, {log_response_ratio_s.max():.3f}]")

    # Create ultra-comprehensive features
    features = create_spatial_features(x_s, y_s)

    # Quantile transformation for robust modeling
    n_quantiles = min(100, len(log_response_ratio_s) // 2)  # More quantiles for better resolution
    qt_s = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=n_quantiles,
        random_state=42
    )
    target_transformed = qt_s.fit_transform(log_response_ratio_s.reshape(-1, 1)).ravel()

    # Train with ultra-high capacity model
    model_s, validation_score = train_lightgbm(
        features, target_transformed, model_type='spatial'
    )

    print(f"Ultra-high-capacity spatial model trained successfully")
    if validation_score is not None:
        print(f"Validation RMSE: {validation_score:.6f}")

    return {
        'model': model_s,
        'quantile_transformer': qt_s,
        'epsilon': epsilon
    }


def train_temporal_model(temporal_data_path):
    """Train temporal model with ULTRA-HIGH-CAPACITY and attention-inspired features.

    Args:
        temporal_data_path (str): Path to temporal data parquet file.

    Returns:
        dict: Dictionary containing trained model and preprocessing artifacts.
    """
    print("\n" + "="*60)
    print("Training ULTRA-HIGH-CAPACITY temporal model with attention mechanism...")

    # Load temporal data
    table_t = pq.read_table(temporal_data_path)
    x_t = table_t["x"].to_numpy()
    y_t = table_t["y"].to_numpy()
    t_t = table_t["t"].to_numpy()  # Time dimension
    nev_t = table_t["nEV"].to_numpy()
    npe_t = table_t["nPE"].to_numpy()

    # Calculate time bin parameters
    t_bins = np.unique(t_t)
    t_binwidth = t_bins[1] - t_bins[0] if len(t_bins) > 1 else 1.0

    print(f"Temporal data: {len(x_t)} points")
    print(f"Time binwidth: {t_binwidth}")

    # Model response ratio with log transform (same as spatial)
    epsilon = 1e-12
    response_ratio_t = (npe_t + epsilon) / (nev_t + epsilon)
    log_response_ratio_t = np.log(response_ratio_t)

    print(f"Response ratio range: [{response_ratio_t.min():.6f}, {response_ratio_t.max():.6f}]")
    print(f"Log response ratio range: [{log_response_ratio_t.min():.3f}, {log_response_ratio_t.max():.3f}]")

    # Create ultra-high-capacity temporal features
    features = create_temporal_features(x_t, y_t, t_t)

    # Quantile transformation (same as spatial)
    n_quantiles_t = min(300, len(log_response_ratio_t) // 2)  # More quantiles
    qt_t = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=n_quantiles_t,
        random_state=42
    )
    target_transformed = qt_t.fit_transform(log_response_ratio_t.reshape(-1, 1)).ravel()

    # Train using ultra-high-capacity LightGBM
    model_t, validation_score = train_lightgbm(
        features, target_transformed, model_type='temporal'
    )

    print(f"ULTRA-HIGH-CAPACITY temporal model trained successfully")
    if validation_score is not None:
        print(f"Validation RMSE: {validation_score:.6f}")

    return {
        'model': model_t,
        'quantile_transformer': qt_t,
        't_binwidth': t_binwidth
    }


def train_lightgbm(features, target, model_type='spatial'):
    """Train LightGBM with ULTRA-HIGH-CAPACITY for extreme variation handling.

    This function creates models with maximum learning capacity to naturally
    handle extreme variations (350x+) without regularization constraints.

    Args:
        features (np.ndarray): Feature matrix.
        target (np.ndarray): Target variable.
        model_type (str): Type of model ('spatial' or 'temporal').

    Returns:
        tuple: Trained model and validation score (if applicable).
    """
    if model_type == 'temporal':
        # Temporal model - MAXIMUM capacity for extreme variation learning
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,           # Very slow learning for precision
            'subsample': 0.95,               # High sampling for maximum data usage
            'subsample_freq': 1,             # Subsample every iteration
            'colsample_bytree': 0.95,        # High feature sampling
            'reg_alpha': 0.001,             # Minimal L1 regularization
            'reg_lambda': 0.01,             # Minimal L2 regularization
            'min_gain_to_split': 0,          # Allow any split that improves
            'min_data_in_leaf': 1,           # Minimal constraint on leaf size
            'random_state': 42,
            'verbose': -1,
            'force_col_wise': True,          # Optimization for many features
            'num_leaves': 1023,          # Maximum leaves for extreme capacity
            'max_depth': 20,             # Very deep trees for complex patterns
            'min_child_samples': 1,      # Allow maximum flexibility
            'min_split_gain': 0,         # No minimum gain requirement
        }
        n_estimators = 10000  # Many trees for comprehensive learning
        print(f"TEMPORAL MODEL: Ultra-high capacity (leaves={params['num_leaves']}, depth={params['max_depth']}, trees={n_estimators})")
    else:
        # Spatial model - use old training params
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'subsample_freq': 5,
            'colsample_bytree': 0.8,
            'min_child_samples': max(5, len(features) // 20),
            'min_split_gain': 0.01,
            'reg_alpha': 0.1,  # L1 regularization
            'reg_lambda': 0.1,  # L2 regularization
            'max_depth': 8,
            'random_state': 42,
            'verbose': -1  # Suppress verbose output
        }
        n_estimators = 2000  # Many trees
        print(f"SPATIAL MODEL: High capacity (leaves={params['num_leaves']}, depth={params['max_depth']}, trees={n_estimators})")

    # Create model instance
    model = lgb.LGBMRegressor(n_estimators=n_estimators, **params)

    # Use validation set if dataset is large enough
    if len(features) > 100:
        x_train, x_val, y_train, y_val = train_test_split(
            features, target, test_size=0.15, random_state=42  # Smaller validation for more training data
        )
        print(f"Using validation: train={len(x_train)}, val={len(x_val)}")

        # Train with early stopping (generous patience for slow learning)
        early_stopping_rounds = 300  # Very high patience for ultra-capacity models
        model.fit(
            x_train, y_train,
            eval_set=[(x_val, y_val)],
            eval_names=['validation'],
            callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(0)]
        )
        validation_score = model.best_score_['validation']['rmse']

        print(f"Final model: {model.best_iteration_} iterations (stopped early)")

    else:
        # Small dataset: use all data for training
        print("Small dataset, using all data for training")
        model.fit(features, target)
        validation_score = None

    return model, validation_score


def save_models(output_path, spatial_data, temporal_data):
    """Save trained ultra-high-capacity models and preprocessing artifacts.

    Args:
        output_path (str): Path for output pickle file.
        spatial_data (dict): Spatial model data.
        temporal_data (dict): Temporal model data.
    """
    # Combine all model artifacts
    model_artifacts = {
        "model_s": spatial_data['model'],
        "model_t": temporal_data['model'],
        "t_binwidth": temporal_data['t_binwidth'],
        "qt_s": spatial_data['quantile_transformer'],
        "qt_t": temporal_data['quantile_transformer'],
        "epsilon": spatial_data['epsilon']
    }

    # Save to pickle file
    with open(output_path, "wb") as file_out:
        pickle.dump(model_artifacts, file_out)

    print(f"\nULTRA-HIGH-CAPACITY models saved to {output_path}")


def main():
    """Main function to execute the ultra-high-capacity training pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train ULTRA-HIGH-CAPACITY GBM models'
    )
    parser.add_argument(
        '-i', dest='input_files', nargs='+', type=str, required=True,
        help='Input parquet files: [spatial_data.parquet, temporal_data.parquet]'
    )
    parser.add_argument(
        '-o', dest='output_file', type=str, required=True,
        help='Output pickle file path for trained models'
    )
    args = parser.parse_args()

    # Validate input arguments
    if len(args.input_files) != 2:
        raise ValueError("Exactly 2 input files required: spatial_data.parquet temporal_data.parquet")

    # Train ultra-high-capacity models
    spatial_model_data = train_spatial_model(args.input_files[0])
    temporal_model_data = train_temporal_model(args.input_files[1])

    # Save combined model artifacts
    save_models(args.output_file, spatial_model_data, temporal_model_data)


if __name__ == "__main__":
    main()
