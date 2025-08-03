"""Robust regression techniques for repeat-sales models"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Callable
from scipy import sparse
from scipy.sparse.linalg import lsqr
from dataclasses import dataclass

from ..algorithms.regression import RepeatSalesRegression, RegressionResults
from .detection import OutlierDetector, OutlierResult


@dataclass
class RobustRegressionConfig:
    """Configuration for robust regression methods"""
    method: str = 'huber'  # 'huber', 'bisquare', 'cauchy', 'welsch'
    tuning_constant: Optional[float] = None  # Auto-determined if None
    max_iterations: int = 50
    convergence_tolerance: float = 1e-4
    outlier_removal: bool = False  # Whether to remove outliers before regression
    outlier_threshold: float = 3.0  # For outlier removal


class RobustRepeatSalesRegression(RepeatSalesRegression):
    """
    Robust repeat-sales regression using M-estimators
    
    This class extends the basic repeat-sales regression with robust
    estimation techniques that downweight outliers automatically.
    """
    
    def __init__(self,
                 config: Optional[RobustRegressionConfig] = None,
                 base_period: Optional[pd.Timestamp] = None,
                 outlier_detector: Optional[OutlierDetector] = None):
        """
        Initialize robust regression
        
        Args:
            config: Configuration for robust methods
            base_period: Base period for index
            outlier_detector: Optional outlier detector for pre-filtering
        """
        super().__init__(base_period=base_period, robust_weights=True)
        self.config = config or RobustRegressionConfig()
        self.outlier_detector = outlier_detector or OutlierDetector()
        self._weights = None
        self._outlier_result = None
        
    def fit(self,
            pairs_df: pd.DataFrame,
            start_date: Optional[pd.Timestamp] = None,
            end_date: Optional[pd.Timestamp] = None) -> RegressionResults:
        """
        Fit robust repeat-sales regression
        
        Args:
            pairs_df: DataFrame of valid transaction pairs
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            
        Returns:
            RegressionResults with robust estimates
        """
        # Filter by date range if specified
        filtered_df = self._filter_date_range(pairs_df, start_date, end_date)
        
        if len(filtered_df) == 0:
            raise ValueError("No transactions in specified date range")
        
        # Optional outlier removal
        if self.config.outlier_removal:
            self._outlier_result = self.outlier_detector.detect_outliers(filtered_df)
            filtered_df = self.outlier_detector.get_clean_data(
                filtered_df, self._outlier_result
            )
            print(f"Removed {len(self._outlier_result.outlier_indices)} outliers")
        
        # Create time periods
        period_dates = self._create_period_index(filtered_df)
        period_map = {date: idx for idx, date in enumerate(period_dates)}
        
        # Map transactions to periods
        filtered_df = self._map_to_periods(filtered_df, period_map)
        
        # Build design matrix
        X, y = self._build_design_matrix(filtered_df, len(period_dates))
        
        # Perform robust regression
        if self.config.method == 'ols':
            # Standard OLS (non-robust)
            coefficients, residuals, convergence = self._perform_regression(X, y)
            weights = np.ones(len(y))
        else:
            # Iteratively reweighted least squares (IRLS)
            coefficients, residuals, weights, convergence = self._perform_robust_regression(X, y)
        
        self._weights = weights
        
        # Calculate robust standard errors
        standard_errors = self._calculate_robust_standard_errors(X, residuals, weights)
        
        # Calculate R-squared (using weighted residuals)
        r_squared = self._calculate_weighted_r_squared(y, residuals, weights)
        
        # Store results
        self.results = RegressionResults(
            log_returns=coefficients,
            standard_errors=standard_errors,
            residuals=residuals,
            r_squared=r_squared,
            num_observations=len(filtered_df),
            num_periods=len(period_dates),
            period_dates=period_dates,
            convergence_info=convergence
        )
        
        return self.results
    
    def _perform_robust_regression(self,
                                 X: sparse.csr_matrix,
                                 y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Perform robust regression using IRLS
        
        Returns:
            Tuple of (coefficients, residuals, weights, convergence_info)
        """
        n, k = X.shape
        
        # Initial OLS estimate
        beta_old = lsqr(X, y, atol=1e-8, btol=1e-8)[0]
        
        # Get weight function and tuning constant
        weight_func, c = self._get_weight_function()
        
        convergence_info = {
            'iterations': 0,
            'converged': False,
            'final_change': np.inf,
            'method': self.config.method,
            'tuning_constant': c
        }
        
        # IRLS iterations
        weights = np.ones(n)  # Initialize weights
        for iteration in range(self.config.max_iterations):
            # Calculate residuals
            residuals = y - X.dot(beta_old)
            
            # Calculate MAD (median absolute deviation) for scaling
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = 1.4826 * mad  # Consistency factor for normal distribution
            
            # Calculate weights
            if scale > 1e-10:
                standardized_resids = residuals / scale
                weights = weight_func(standardized_resids, c)
            else:
                weights = np.ones(n)
            
            # Weighted least squares
            W = sparse.diags(np.sqrt(weights))
            X_weighted = W.dot(X)
            y_weighted = W.dot(y)
            
            # Solve weighted system
            beta_new = lsqr(X_weighted, y_weighted, atol=1e-8, btol=1e-8)[0]
            
            # Check convergence
            change = np.max(np.abs(beta_new - beta_old))
            convergence_info['final_change'] = change
            convergence_info['iterations'] = iteration + 1
            
            if change < self.config.convergence_tolerance:
                convergence_info['converged'] = True
                break
            
            beta_old = beta_new
        
        # Final residuals
        residuals = y - X.dot(beta_new)
        
        # Add zero for base period
        coefficients = np.insert(beta_new, 0, 0.0)
        
        return coefficients, residuals, weights, convergence_info
    
    def _get_weight_function(self) -> Tuple[Callable, float]:
        """Get weight function and tuning constant for robust method"""
        
        if self.config.method == 'huber':
            c = self.config.tuning_constant or 1.345
            def weight_func(r, c):
                return np.where(np.abs(r) <= c, 1.0, c / np.abs(r))
            
        elif self.config.method == 'bisquare':
            c = self.config.tuning_constant or 4.685
            def weight_func(r, c):
                return np.where(np.abs(r) <= c, (1 - (r/c)**2)**2, 0.0)
            
        elif self.config.method == 'cauchy':
            c = self.config.tuning_constant or 2.385
            def weight_func(r, c):
                return 1 / (1 + (r/c)**2)
            
        elif self.config.method == 'welsch':
            c = self.config.tuning_constant or 2.985
            def weight_func(r, c):
                return np.exp(-(r/c)**2)
            
        else:
            raise ValueError(f"Unknown robust method: {self.config.method}")
        
        return weight_func, c
    
    def _calculate_robust_standard_errors(self,
                                        X: sparse.csr_matrix,
                                        residuals: np.ndarray,
                                        weights: np.ndarray) -> np.ndarray:
        """Calculate robust standard errors accounting for weights"""
        n, k = X.shape
        
        # Weighted residual variance
        weighted_residuals = residuals * np.sqrt(weights)
        sigma_squared = np.sum(weighted_residuals**2) / (n - k)
        
        # Weighted X'X
        W = sparse.diags(weights)
        XtWX = X.T.dot(W).dot(X)
        
        try:
            XtWX_inv = np.linalg.inv(XtWX.toarray())
            
            # Simple robust standard errors: sqrt(diag((X'WX)^-1) * sigma^2)
            # This is more stable than the sandwich estimator for small samples
            variance_diag = np.diag(XtWX_inv) * sigma_squared
            
            # Ensure all diagonal elements are non-negative
            variance_diag = np.maximum(variance_diag, 0)
            
            standard_errors = np.sqrt(variance_diag)
            
        except np.linalg.LinAlgError:
            # If inversion fails, use simple approximation
            # Standard errors proportional to 1/sqrt(sum of weights for each coefficient)
            standard_errors = np.zeros(k)
            for j in range(k):
                col_weights = np.abs(X[:, j].toarray().flatten()) * weights
                weight_sum = np.sum(col_weights)
                if weight_sum > 0:
                    standard_errors[j] = sigma_squared / np.sqrt(weight_sum)
                else:
                    standard_errors[j] = np.nan
        
        # Add zero for base period
        standard_errors = np.insert(standard_errors, 0, 0.0)
        
        return standard_errors
    
    def _calculate_weighted_r_squared(self,
                                    y: np.ndarray,
                                    residuals: np.ndarray,
                                    weights: np.ndarray) -> float:
        """Calculate R-squared with weights"""
        # Weighted sum of squares
        weighted_residuals = residuals * np.sqrt(weights)
        ss_res = np.sum(weighted_residuals**2)
        
        # Weighted total sum of squares
        y_weighted_mean = np.sum(y * weights) / np.sum(weights)
        weighted_deviations = (y - y_weighted_mean) * np.sqrt(weights)
        ss_tot = np.sum(weighted_deviations**2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def get_weights(self) -> Optional[np.ndarray]:
        """Get the robust weights from the last fit"""
        return self._weights
    
    def get_outlier_result(self) -> Optional[OutlierResult]:
        """Get outlier detection results if outlier removal was used"""
        return self._outlier_result
    
    def get_influence_statistics(self) -> pd.DataFrame:
        """
        Get influence statistics for each observation
        
        Returns:
            DataFrame with weights and influence measures
        """
        if self._weights is None:
            raise ValueError("Must fit model before getting influence statistics")
        
        n = len(self._weights)
        
        # Create influence DataFrame
        influence_df = pd.DataFrame({
            'observation': range(n),
            'weight': self._weights,
            'residual': self.results.residuals,
            'standardized_residual': self.results.residuals / np.std(self.results.residuals),
            'downweighted': self._weights < 0.5
        })
        
        # Add percentile ranks
        influence_df['weight_percentile'] = influence_df['weight'].rank(pct=True)
        
        return influence_df
    
    def compare_with_ols(self,
                        pairs_df: pd.DataFrame,
                        start_date: Optional[pd.Timestamp] = None,
                        end_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Compare robust results with standard OLS
        
        Args:
            pairs_df: DataFrame of transaction pairs
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            DataFrame comparing robust and OLS estimates
        """
        # Get robust results (already fitted)
        robust_results = self.results
        
        # Fit standard OLS
        ols_regression = RepeatSalesRegression(
            base_period=self.base_period,
            robust_weights=False
        )
        ols_results = ols_regression.fit(pairs_df, start_date, end_date)
        
        # Compare coefficients
        comparison_data = []
        for i, date in enumerate(robust_results.period_dates):
            comparison_data.append({
                'date': date,
                'robust_coefficient': robust_results.log_returns[i],
                'ols_coefficient': ols_results.log_returns[i],
                'difference': robust_results.log_returns[i] - ols_results.log_returns[i],
                'robust_se': robust_results.standard_errors[i],
                'ols_se': ols_results.standard_errors[i]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Add index values
        comparison_df['robust_index'] = 100 * np.exp(comparison_df['robust_coefficient'])
        comparison_df['ols_index'] = 100 * np.exp(comparison_df['ols_coefficient'])
        comparison_df['index_difference'] = comparison_df['robust_index'] - comparison_df['ols_index']
        
        return comparison_df