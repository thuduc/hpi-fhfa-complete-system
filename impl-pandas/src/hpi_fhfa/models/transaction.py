"""Transaction data models"""

from dataclasses import dataclass
from datetime import date
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class TransactionPair:
    """Represents a repeat-sales transaction pair"""
    property_id: str
    tract_id: str
    cbsa_id: str
    first_sale_date: date
    first_sale_price: float
    second_sale_date: date
    second_sale_price: float
    
    @property
    def log_price_diff(self) -> float:
        """Calculate log price difference"""
        return np.log(self.second_sale_price) - np.log(self.first_sale_price)
    
    @property
    def time_diff_years(self) -> float:
        """Calculate time difference in years"""
        return (self.second_sale_date - self.first_sale_date).days / 365.25
    
    @property
    def cagr(self) -> float:
        """Calculate compound annual growth rate"""
        if self.time_diff_years == 0:
            return 0.0
        return (self.second_sale_price / self.first_sale_price) ** (1 / self.time_diff_years) - 1


def validate_transaction_pair(pair: TransactionPair) -> tuple[bool, Optional[str]]:
    """
    Validate a transaction pair against quality filters
    
    Returns:
        (is_valid, error_message)
    """
    # Check for same 12-month period
    months_diff = (pair.second_sale_date.year - pair.first_sale_date.year) * 12 + \
                  (pair.second_sale_date.month - pair.first_sale_date.month)
    
    if abs(months_diff) < 12:
        return False, "Transactions within same 12-month period"
    
    # Check CAGR bounds
    cagr = pair.cagr
    if abs(cagr) > 0.30:
        return False, f"CAGR {cagr:.2%} exceeds ±30% threshold"
    
    # Check cumulative appreciation
    appreciation = pair.second_sale_price / pair.first_sale_price
    if appreciation > 10.0:
        return False, f"Appreciation {appreciation:.2f}x exceeds 10x threshold"
    if appreciation < 0.75:
        return False, f"Appreciation {appreciation:.2f}x below 0.75x threshold"
    
    # Check for positive prices
    if pair.first_sale_price <= 0 or pair.second_sale_price <= 0:
        return False, "Invalid price (must be positive)"
    
    # Check dates
    if pair.second_sale_date <= pair.first_sale_date:
        return False, "Second sale must be after first sale"
    
    return True, None


def create_transaction_pairs_df(pairs: list[TransactionPair]) -> pd.DataFrame:
    """Convert list of TransactionPair objects to DataFrame"""
    data = []
    for pair in pairs:
        is_valid, _ = validate_transaction_pair(pair)
        if is_valid:
            data.append({
                'property_id': pair.property_id,
                'tract_id': pair.tract_id,
                'cbsa_id': pair.cbsa_id,
                'first_sale_date': pair.first_sale_date,
                'first_sale_price': pair.first_sale_price,
                'second_sale_date': pair.second_sale_date,
                'second_sale_price': pair.second_sale_price,
                'log_price_diff': pair.log_price_diff,
                'time_diff_years': pair.time_diff_years,
                'cagr': pair.cagr
            })
    
    return pd.DataFrame(data)