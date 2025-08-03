"""Bailey-Muth-Nourse Repeat-Sales Regression Implementation"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from datetime import date

from ..models.transaction import TransactionPair


@dataclass
class RegressionResults:
    """Results from repeat-sales regression"""
    log_returns: np.ndarray
    standard_errors: np.ndarray
    residuals: np.ndarray
    r_squared: float
    num_observations: int
    num_periods: int
    period_dates: List[date]
    convergence_info: Dict[str, any]


class RepeatSalesRegression:
    """
    Bailey-Muth-Nourse (BMN) repeat-sales regression implementation
    
    The BMN method estimates housing price indices by regressing
    log price changes on time dummy variables.
    """
    
    def __init__(self, 
                 base_period: Optional[date] = None,
                 robust_weights: bool = True):
        """
        Initialize regression model
        
        Args:
            base_period: Base period for index (default: first period)
            robust_weights: Use robust weighting scheme to downweight outliers
        """
        self.base_period = base_period
        self.robust_weights = robust_weights
        self.results = None
        
    def fit(self, 
            pairs_df: pd.DataFrame,
            start_date: Optional[date] = None,
            end_date: Optional[date] = None) -> RegressionResults:
        """
        Fit repeat-sales regression model
        
        Args:
            pairs_df: DataFrame of valid transaction pairs
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            
        Returns:
            RegressionResults object
        """
        # Filter by date range if specified
        filtered_df = self._filter_date_range(pairs_df, start_date, end_date)
        
        if len(filtered_df) == 0:
            raise ValueError("No transactions in specified date range")
        
        # Create time periods (monthly)
        period_dates = self._create_period_index(filtered_df)
        period_map = {date: idx for idx, date in enumerate(period_dates)}
        
        # Map transactions to periods
        filtered_df = self._map_to_periods(filtered_df, period_map)
        
        # Build design matrix
        X, y = self._build_design_matrix(filtered_df, len(period_dates))
        
        # Perform regression
        coefficients, residuals, convergence = self._perform_regression(X, y)
        
        # Calculate standard errors
        standard_errors = self._calculate_standard_errors(X, residuals)
        
        # Calculate R-squared
        r_squared = self._calculate_r_squared(y, residuals)
        
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
    
    def _filter_date_range(self,
                          pairs_df: pd.DataFrame,
                          start_date: Optional[date],
                          end_date: Optional[date]) -> pd.DataFrame:
        """Filter pairs by date range"""
        df = pairs_df.copy()
        
        # Ensure date columns are datetime
        for col in ['first_sale_date', 'second_sale_date']:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        
        if start_date:
            df = df[df['second_sale_date'] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df['second_sale_date'] <= pd.Timestamp(end_date)]
            
        return df
    
    def _create_period_index(self, pairs_df: pd.DataFrame) -> List[date]:
        """Create monthly period index from transaction dates"""
        # Get all unique year-month combinations
        all_dates = pd.concat([
            pd.to_datetime(pairs_df['first_sale_date']),
            pd.to_datetime(pairs_df['second_sale_date'])
        ])
        
        # Create monthly periods
        min_date = all_dates.min()
        max_date = all_dates.max()
        
        period_range = pd.date_range(
            start=min_date.replace(day=1),
            end=max_date.replace(day=1),
            freq='MS'  # Month start
        )
        
        return [d.date() for d in period_range]
    
    def _map_to_periods(self,
                       pairs_df: pd.DataFrame,
                       period_map: Dict[date, int]) -> pd.DataFrame:
        """Map transaction dates to period indices"""
        df = pairs_df.copy()
        
        # Convert to period indices
        df['first_period'] = pd.to_datetime(df['first_sale_date']).dt.to_period('M').dt.to_timestamp().dt.date
        df['second_period'] = pd.to_datetime(df['second_sale_date']).dt.to_period('M').dt.to_timestamp().dt.date
        
        df['first_period_idx'] = df['first_period'].map(period_map)
        df['second_period_idx'] = df['second_period'].map(period_map)
        
        # Remove any transactions with unmapped periods
        df = df.dropna(subset=['first_period_idx', 'second_period_idx'])
        df['first_period_idx'] = df['first_period_idx'].astype(int)
        df['second_period_idx'] = df['second_period_idx'].astype(int)
        
        return df
    
    def _build_design_matrix(self,
                           pairs_df: pd.DataFrame,
                           num_periods: int) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Build sparse design matrix for regression
        
        The design matrix has one row per transaction pair and one column
        per time period. Entry (i,t) is:
        - +1 if property i sold in period t (second sale)
        - -1 if property i sold in period t (first sale)
        - 0 otherwise
        """
        n_obs = len(pairs_df)
        
        # Build sparse matrix efficiently
        row_indices = []
        col_indices = []
        data = []
        
        for idx, (_, row) in enumerate(pairs_df.iterrows()):
            # First sale: -1
            row_indices.append(idx)
            col_indices.append(row['first_period_idx'])
            data.append(-1)
            
            # Second sale: +1
            row_indices.append(idx)
            col_indices.append(row['second_period_idx'])
            data.append(1)
        
        # Create sparse matrix
        X = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(n_obs, num_periods)
        )
        
        # Create log price ratio vector
        y = np.log(pairs_df['second_sale_price'].values / pairs_df['first_sale_price'].values)
        
        # Drop base period (first column) to avoid multicollinearity
        if self.base_period is None:
            X = X[:, 1:]
        else:
            # Find base period index and drop it
            # For now, just drop first period
            X = X[:, 1:]
        
        return X, y
    
    def _perform_regression(self,
                          X: sparse.csr_matrix,
                          y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Perform the actual regression using sparse least squares
        
        Returns coefficients, residuals, and convergence info
        """
        # Use LSQR for sparse least squares
        results = lsqr(X, y, atol=1e-8, btol=1e-8, iter_lim=1000)
        
        coefficients = results[0]
        residuals = y - X.dot(coefficients)
        
        # Add zero for base period
        coefficients = np.insert(coefficients, 0, 0.0)
        
        convergence_info = {
            'iterations': results[2],
            'converged': results[1] == 1,
            'condition_number': results[6],
            'norm_residual': results[3]
        }
        
        return coefficients, residuals, convergence_info
    
    def _calculate_standard_errors(self,
                                 X: sparse.csr_matrix,
                                 residuals: np.ndarray) -> np.ndarray:
        """Calculate standard errors of coefficients"""
        n = X.shape[0]
        k = X.shape[1]
        
        # Calculate residual variance
        sigma_squared = np.dot(residuals, residuals) / (n - k)
        
        # Calculate (X'X)^-1 for variance-covariance matrix
        # For large sparse matrices, use approximation
        XtX = X.T.dot(X)
        
        # Add small regularization for numerical stability
        XtX_dense = XtX.toarray().astype(np.float64)
        XtX_dense += np.eye(k) * 1e-10
        
        try:
            XtX_inv = np.linalg.inv(XtX_dense)
            variances = np.diag(XtX_inv) * sigma_squared
            standard_errors = np.sqrt(variances)
        except np.linalg.LinAlgError:
            # If singular, return NaN
            standard_errors = np.full(k, np.nan)
        
        # Add zero for base period
        standard_errors = np.insert(standard_errors, 0, 0.0)
        
        return standard_errors
    
    def _calculate_r_squared(self, y: np.ndarray, residuals: np.ndarray) -> float:
        """Calculate R-squared statistic"""
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def get_index_values(self, base_value: float = 100.0) -> pd.DataFrame:
        """
        Convert log returns to index values
        
        Args:
            base_value: Base index value (default 100)
            
        Returns:
            DataFrame with dates and index values
        """
        if self.results is None:
            raise ValueError("Must fit model before getting index values")
        
        # Convert log returns to index levels
        index_values = base_value * np.exp(self.results.log_returns)
        
        return pd.DataFrame({
            'date': self.results.period_dates,
            'index_value': index_values,
            'log_return': self.results.log_returns,
            'standard_error': self.results.standard_errors
        })
    
    def calculate_returns(self, 
                         period_length: str = 'quarterly') -> pd.DataFrame:
        """
        Calculate period-over-period returns
        
        Args:
            period_length: 'monthly', 'quarterly', or 'annual'
            
        Returns:
            DataFrame with period returns
        """
        if self.results is None:
            raise ValueError("Must fit model before calculating returns")
        
        index_df = self.get_index_values()
        
        if period_length == 'monthly':
            periods = 1
        elif period_length == 'quarterly':
            periods = 3
        elif period_length == 'annual':
            periods = 12
        else:
            raise ValueError(f"Invalid period length: {period_length}")
        
        index_df['return'] = index_df['index_value'].pct_change(periods)
        
        return index_df[['date', 'return']].dropna()