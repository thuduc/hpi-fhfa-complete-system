"""Outlier detection algorithms for repeat-sales data"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from scipy import stats
from scipy.sparse import csr_matrix

from ..models.transaction import TransactionPair
from ..algorithms.regression import RegressionResults


@dataclass
class OutlierResult:
    """Results from outlier detection"""
    outlier_indices: Set[int]  # Indices of detected outliers
    outlier_scores: Dict[int, float]  # Index -> outlier score
    outlier_reasons: Dict[int, List[str]]  # Index -> list of reasons
    statistics: Dict[str, float]  # Summary statistics
    thresholds: Dict[str, float]  # Thresholds used


class OutlierDetector:
    """
    Detect outliers in repeat-sales data using multiple methods
    
    Methods include:
    - Cook's distance: Influence of each observation on regression
    - Leverage: Unusual predictor values
    - Studentized residuals: Standardized residuals
    - CAGR outliers: Extreme price appreciation
    - Time gap outliers: Very long holding periods
    """
    
    def __init__(self,
                 cooks_threshold: float = 4.0,
                 leverage_threshold: float = 3.0,
                 residual_threshold: float = 3.0,
                 cagr_threshold: float = 0.5,  # 50% annual appreciation
                 time_gap_years: float = 30.0):
        """
        Initialize outlier detector
        
        Args:
            cooks_threshold: Threshold for Cook's distance (times mean)
            leverage_threshold: Threshold for leverage (times mean)
            residual_threshold: Threshold for studentized residuals (std devs)
            cagr_threshold: Maximum allowed CAGR
            time_gap_years: Maximum allowed holding period
        """
        self.cooks_threshold = cooks_threshold
        self.leverage_threshold = leverage_threshold
        self.residual_threshold = residual_threshold
        self.cagr_threshold = cagr_threshold
        self.time_gap_years = time_gap_years
        
    def detect_outliers(self,
                       pairs_df: pd.DataFrame,
                       regression_results: Optional[RegressionResults] = None,
                       design_matrix: Optional[csr_matrix] = None) -> OutlierResult:
        """
        Detect outliers using multiple methods
        
        Args:
            pairs_df: DataFrame of transaction pairs
            regression_results: Optional regression results for statistical outliers
            design_matrix: Optional design matrix X for leverage calculation
            
        Returns:
            OutlierResult with detected outliers and reasons
        """
        outlier_indices = set()
        outlier_scores = {}
        outlier_reasons = {}
        
        # 1. CAGR outliers (can be done without regression)
        cagr_outliers = self._detect_cagr_outliers(pairs_df)
        outlier_indices.update(cagr_outliers)
        for idx in cagr_outliers:
            outlier_reasons.setdefault(idx, []).append("Extreme CAGR")
            outlier_scores[idx] = outlier_scores.get(idx, 0) + 1
        
        # 2. Time gap outliers
        time_outliers = self._detect_time_gap_outliers(pairs_df)
        outlier_indices.update(time_outliers)
        for idx in time_outliers:
            outlier_reasons.setdefault(idx, []).append("Excessive time gap")
            outlier_scores[idx] = outlier_scores.get(idx, 0) + 1
        
        # Statistical outliers (require regression results)
        cooks_outliers = set()
        residual_outliers = set()
        leverage_outliers = set()
        
        if regression_results is not None:
            # 3. Cook's distance
            if design_matrix is not None:
                cooks_outliers, cooks_d = self._detect_cooks_distance_outliers(
                    regression_results, design_matrix
                )
                outlier_indices.update(cooks_outliers)
                for idx in cooks_outliers:
                    outlier_reasons.setdefault(idx, []).append("High Cook's distance")
                    outlier_scores[idx] = outlier_scores.get(idx, 0) + cooks_d[idx]
            
            # 4. Studentized residuals
            residual_outliers, student_resids = self._detect_residual_outliers(
                regression_results
            )
            outlier_indices.update(residual_outliers)
            for idx in residual_outliers:
                outlier_reasons.setdefault(idx, []).append("Large studentized residual")
                outlier_scores[idx] = outlier_scores.get(idx, 0) + abs(student_resids[idx])
            
            # 5. Leverage points
            if design_matrix is not None:
                leverage_outliers, leverages = self._detect_leverage_outliers(
                    design_matrix
                )
                outlier_indices.update(leverage_outliers)
                for idx in leverage_outliers:
                    outlier_reasons.setdefault(idx, []).append("High leverage")
                    outlier_scores[idx] = outlier_scores.get(idx, 0) + leverages[idx]
        
        # Calculate statistics
        statistics = {
            'total_observations': len(pairs_df),
            'total_outliers': len(outlier_indices),
            'outlier_rate': len(outlier_indices) / len(pairs_df) if len(pairs_df) > 0 else 0,
            'cagr_outliers': len(cagr_outliers),
            'time_gap_outliers': len(time_outliers)
        }
        
        if regression_results:
            statistics.update({
                'cooks_outliers': len(cooks_outliers),
                'residual_outliers': len(residual_outliers),
                'leverage_outliers': len(leverage_outliers)
            })
        
        thresholds = {
            'cooks_distance': self.cooks_threshold,
            'leverage': self.leverage_threshold,
            'studentized_residual': self.residual_threshold,
            'cagr': self.cagr_threshold,
            'time_gap_years': self.time_gap_years
        }
        
        return OutlierResult(
            outlier_indices=outlier_indices,
            outlier_scores=outlier_scores,
            outlier_reasons=outlier_reasons,
            statistics=statistics,
            thresholds=thresholds
        )
    
    def _detect_cagr_outliers(self, pairs_df: pd.DataFrame) -> Set[int]:
        """Detect extreme CAGR outliers"""
        outliers = set()
        
        for idx, row in pairs_df.iterrows():
            # Calculate CAGR
            years = (pd.to_datetime(row['second_sale_date']) - 
                    pd.to_datetime(row['first_sale_date'])).days / 365.25
            
            if years > 0:
                cagr = (row['second_sale_price'] / row['first_sale_price']) ** (1/years) - 1
                
                # Check both positive and negative extremes
                if abs(cagr) > self.cagr_threshold:
                    outliers.add(idx)
        
        return outliers
    
    def _detect_time_gap_outliers(self, pairs_df: pd.DataFrame) -> Set[int]:
        """Detect excessive time gaps"""
        outliers = set()
        
        for idx, row in pairs_df.iterrows():
            years = (pd.to_datetime(row['second_sale_date']) - 
                    pd.to_datetime(row['first_sale_date'])).days / 365.25
            
            if years > self.time_gap_years:
                outliers.add(idx)
        
        return outliers
    
    def _detect_cooks_distance_outliers(self,
                                      regression_results: RegressionResults,
                                      X: csr_matrix) -> Tuple[Set[int], Dict[int, float]]:
        """
        Detect outliers using Cook's distance
        
        Cook's distance measures the influence of each observation on the
        regression coefficients.
        """
        n = X.shape[0]
        k = X.shape[1]
        
        # Calculate hat matrix diagonal elements (leverage)
        # For large sparse matrices, approximate using diagonal of X(X'X)^-1X'
        XtX = X.T.dot(X)
        try:
            XtX_inv = np.linalg.inv(XtX.toarray())
        except np.linalg.LinAlgError:
            # If singular, return empty results
            return set(), {}
        
        cooks_distances = {}
        outliers = set()
        
        # Calculate Cook's distance for each observation
        for i in range(n):
            # Get the ith row of X
            xi = X.getrow(i).toarray().flatten()
            
            # Leverage for observation i
            hi = xi.dot(XtX_inv).dot(xi.T)
            
            # Studentized residual
            residual = regression_results.residuals[i]
            mse = np.sum(regression_results.residuals**2) / (n - k)
            student_resid = residual / (np.sqrt(mse * (1 - hi)) + 1e-8)
            
            # Cook's distance
            cooks_d = (student_resid**2 * hi) / (k * (1 - hi + 1e-8))
            cooks_distances[i] = cooks_d
            
            # Check threshold (common rule: 4/n or specified threshold times mean)
            if len(cooks_distances) > 0:
                threshold = max(4/n, self.cooks_threshold * np.mean(list(cooks_distances.values())))
                if cooks_d > threshold:
                    outliers.add(i)
        
        return outliers, cooks_distances
    
    def _detect_residual_outliers(self,
                                regression_results: RegressionResults) -> Tuple[Set[int], Dict[int, float]]:
        """Detect outliers using studentized residuals"""
        residuals = regression_results.residuals
        n = len(residuals)
        
        # Calculate studentized residuals
        # Standard error of residuals
        k = regression_results.num_periods - 1  # Degrees of freedom
        mse = np.sum(residuals**2) / (n - k)
        se_resid = np.sqrt(mse)
        
        # Studentize
        studentized = residuals / (se_resid + 1e-8)
        
        outliers = set()
        student_dict = {}
        
        for i, student_resid in enumerate(studentized):
            student_dict[i] = student_resid
            if abs(student_resid) > self.residual_threshold:
                outliers.add(i)
        
        return outliers, student_dict
    
    def _detect_leverage_outliers(self, X: csr_matrix) -> Tuple[Set[int], Dict[int, float]]:
        """Detect high leverage points"""
        n, k = X.shape
        
        # Calculate leverage (diagonal of hat matrix)
        XtX = X.T.dot(X)
        try:
            XtX_inv = np.linalg.inv(XtX.toarray())
        except np.linalg.LinAlgError:
            # If singular, return empty results
            return set(), {}
        
        leverages = {}
        outliers = set()
        
        for i in range(n):
            xi = X.getrow(i).toarray().flatten()
            hi = xi.dot(XtX_inv).dot(xi.T)
            leverages[i] = hi
            
            # Common threshold: 3 * k/n
            threshold = self.leverage_threshold * k / n
            if hi > threshold:
                outliers.add(i)
        
        return outliers, leverages
    
    def get_clean_data(self,
                      pairs_df: pd.DataFrame,
                      outlier_result: OutlierResult) -> pd.DataFrame:
        """
        Return dataset with outliers removed
        
        Args:
            pairs_df: Original pairs DataFrame
            outlier_result: Results from outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        clean_indices = set(range(len(pairs_df))) - outlier_result.outlier_indices
        return pairs_df.iloc[list(clean_indices)]
    
    def flag_outliers(self,
                     pairs_df: pd.DataFrame,
                     outlier_result: OutlierResult) -> pd.DataFrame:
        """
        Add outlier flags and scores to DataFrame
        
        Args:
            pairs_df: Original pairs DataFrame
            outlier_result: Results from outlier detection
            
        Returns:
            DataFrame with outlier flags and scores added
        """
        df = pairs_df.copy()
        
        # Add outlier flag
        df['is_outlier'] = False
        df.loc[list(outlier_result.outlier_indices), 'is_outlier'] = True
        
        # Add outlier score
        df['outlier_score'] = 0.0
        for idx, score in outlier_result.outlier_scores.items():
            if idx in df.index:
                df.loc[idx, 'outlier_score'] = score
        
        # Add outlier reasons
        df['outlier_reasons'] = ''
        for idx, reasons in outlier_result.outlier_reasons.items():
            if idx in df.index:
                df.loc[idx, 'outlier_reasons'] = '; '.join(reasons)
        
        return df
    
    def summarize_outliers(self, outlier_result: OutlierResult) -> pd.DataFrame:
        """
        Create summary report of outliers
        
        Args:
            outlier_result: Results from outlier detection
            
        Returns:
            DataFrame with outlier summary
        """
        # Count reasons
        reason_counts = {}
        for reasons in outlier_result.outlier_reasons.values():
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Create summary
        summary_data = []
        
        # Overall statistics
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Total Observations',
            'Value': outlier_result.statistics['total_observations']
        })
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Total Outliers',
            'Value': outlier_result.statistics['total_outliers']
        })
        summary_data.append({
            'Category': 'Overall',
            'Metric': 'Outlier Rate',
            'Value': f"{outlier_result.statistics['outlier_rate']:.2%}"
        })
        
        # Reason breakdown
        for reason, count in reason_counts.items():
            summary_data.append({
                'Category': 'By Reason',
                'Metric': reason,
                'Value': count
            })
        
        # Threshold settings
        for threshold_name, threshold_value in outlier_result.thresholds.items():
            summary_data.append({
                'Category': 'Thresholds',
                'Metric': threshold_name,
                'Value': threshold_value
            })
        
        return pd.DataFrame(summary_data)