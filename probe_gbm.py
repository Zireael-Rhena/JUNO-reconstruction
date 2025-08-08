#!/usr/bin/env python3
"""Ultra-High-Capacity GBM-based probe with attention-inspired features.

This module implements a Probe class that uses trained LightGBM models
with attention-inspired features and ultra-high capacity designed for
handling extreme variations (350x+) without geometric assumptions.

The approach leverages:
- Attention-inspired feature learning without hardcoded regions
- Ultra-high model capacity for natural extreme variation learning
- Multi-scale universal features for comprehensive pattern capture
- Data-driven pattern recognition rather than assumption-based modeling
"""

import numpy as np
import pickle
from coefficient import ProbeBase
from config import *

import gbm


class Probe(ProbeBase):
    """Ultra-High-Capacity GBM-based probe with attention-inspired modeling.

    This class loads pre-trained spatial and temporal models that use
    attention-inspired features and ultra-high capacity to naturally
    handle extreme temporal response variations without geometric assumptions.

    Key innovations:
    - No hardcoded spatial thresholds or region boundaries
    - Attention mechanism for adaptive feature focus
    - Ultra-high model capacity (1023 leaves, depth 20, 8000 trees)
    - Multi-scale universal feature learning
    - Data-driven extreme variation handling
    """

    def __init__(self, fn):
        """Initialize the probe with ultra-high-capacity trained models.

        Args:
            fn (str): Path to pickle file containing trained model artifacts.
        """
        with open(fn, "rb") as fin:
            models = pickle.load(fin)
        self.model_s = models["model_s"]
        self.model_t = models["model_t"]
        self.t_binwidth = models["t_binwidth"]
        self.qt_s = models["qt_s"]
        self.qt_t = models["qt_t"]
        self.epsilon = models["epsilon"]
        self.debug_count = 0

    def _build_spatial_features(self, rs, thetas):
        """Build ultra-comprehensive spatial features with attention mechanism.

        Args:
            rs (np.ndarray): Radial distances from center.
            thetas (np.ndarray): Angular coordinates in radians.

        Returns:
            np.ndarray: Feature matrix with ultra-comprehensive spatial features.
        """
        # Convert to Cartesian coordinates for gbm.create_spatial_features
        xs = rs * np.cos(thetas)
        ys = rs * np.sin(thetas)

        # Debug information for first call
        if self.debug_count == 0:
            print(f"rs range: [{rs.min():.1f}, {rs.max():.1f}]")
            print(f"thetas range: [{thetas.min():.3f}, {thetas.max():.3f}]")

        return gbm.create_spatial_features(xs, ys)

    def _build_temporal_features(self, rs, thetas, ts):
        """Build ULTRA-HIGH-CAPACITY temporal features with attention mechanism.

        Args:
            rs (np.ndarray): Radial distances from center.
            thetas (np.ndarray): Angular coordinates in radians.
            ts (np.ndarray): Time coordinates.

        Returns:
            np.ndarray: Feature matrix with ultra-high-capacity temporal features.
        """
        # Convert to Cartesian coordinates
        xs = rs * np.cos(thetas)
        ys = rs * np.sin(thetas)
        
        # Debug information for first call
        if self.debug_count == 0:
            print(f"ts range: [{ts.min():.1f}, {ts.max():.1f}]")

        return gbm.create_temporal_features(xs, ys, ts)

    def get_mu(self, rs, thetas):
        """Get spatial response ratio using ultra-comprehensive universal features.

        Args:
            rs (np.ndarray): Radial distances from detector center.
            thetas (np.ndarray): Angular coordinates in radians.

        Returns:
            np.ndarray: Predicted response ratios normalized by FACTOR.
        """
        # Handle angle wrapping for values > pi
        thetas = np.array(thetas)
        thetas[np.where(thetas > np.pi)] = 2*np.pi - thetas[np.where(thetas > np.pi)]

        shape = rs.shape
        rs_flat = rs.flatten()
        thetas_flat = thetas.flatten()

        # Build ultra-comprehensive spatial features
        X = self._build_spatial_features(rs_flat, thetas_flat)

        # Make prediction using ultra-high-capacity spatial model
        pred_transformed = self.model_s.predict(X)

        if self.debug_count == 0:
            print(f"pred_transformed range: [{pred_transformed.min():.6f}, {pred_transformed.max():.6f}]")

        # Inverse transform from quantile-normalized space
        log_response_ratio = self.qt_s.inverse_transform(pred_transformed.reshape(-1, 1)).ravel()

        if self.debug_count == 0:
            print(f"log_response_ratio range: [{log_response_ratio.min():.3f}, {log_response_ratio.max():.3f}]")

        # Convert from log space to response ratio
        response_ratio = np.exp(log_response_ratio)

        if self.debug_count == 0:
            print(f"response_ratio (before clip): [{response_ratio.min():.6f}, {response_ratio.max():.6f}]")

        # Conservative clipping for extreme variations
        response_ratio = np.clip(response_ratio, 1e-10, 1e10)  # Very wide range for robustness

        if self.debug_count == 0:
            print(f"response_ratio (after clip): [{response_ratio.min():.6f}, {response_ratio.max():.6f}]")
            print(f"FACTOR normalization: {FACTOR}")

        result = response_ratio.reshape(shape)
        final_result = result / FACTOR

        if self.debug_count == 0:
            print(f"final spatial result: [{final_result.min():.8f}, {final_result.max():.8f}]")
            print(f"Ultra-comprehensive spatial model with {X.shape[1]} attention-inspired features")

        return final_result

    def get_lc(self, rs, thetas, ts):
        """Get temporal response using ULTRA-HIGH-CAPACITY attention mechanism.

        Args:
            rs (np.ndarray): Radial distances from detector center.
            thetas (np.ndarray): Angular coordinates in radians.
            ts (np.ndarray): Time coordinates.

        Returns:
            np.ndarray: Predicted temporal response ratios normalized by FACTOR and time binwidth.
        """
        # Handle angle wrapping for values > pi
        thetas = np.array(thetas)
        thetas[np.where(thetas > np.pi)] = 2*np.pi - thetas[np.where(thetas > np.pi)]

        shape = rs.shape
        rs_flat = rs.flatten()
        thetas_flat = thetas.flatten()
        ts_flat = ts.flatten()

        # Build ULTRA-HIGH-CAPACITY temporal features
        X = self._build_temporal_features(rs_flat, thetas_flat, ts_flat)

        # Make prediction using ultra-high-capacity temporal model
        pred_transformed = self.model_t.predict(X)

        if self.debug_count == 0:
            print(f"temporal pred_transformed: [{pred_transformed.min():.6f}, {pred_transformed.max():.6f}]")

        # Inverse transform from quantile-normalized space (same as spatial)
        log_response_ratio = self.qt_t.inverse_transform(pred_transformed.reshape(-1, 1)).ravel()

        if self.debug_count == 0:
            print(f"temporal log_response_ratio: [{log_response_ratio.min():.3f}, {log_response_ratio.max():.3f}]")

        # Convert from log space to response ratio (same as spatial)
        response_ratio = np.exp(log_response_ratio)

        # Robust clipping for extreme temporal variations
        response_ratio = np.clip(response_ratio, 1e-10, 1e10)  # Handle extreme variations

        if self.debug_count == 0:
            self.debug_count += 1

        result = response_ratio.reshape(shape)
        return result / FACTOR / self.t_binwidth
