"""Generate synthetic housing transaction data for testing"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import List, Dict, Tuple, Optional
import random
from dataclasses import dataclass


@dataclass
class MarketProfile:
    """Profile for generating realistic market data"""
    base_price_mean: float
    base_price_std: float
    annual_appreciation_mean: float
    annual_appreciation_std: float
    volatility: float
    transaction_rate: float  # Annual probability of sale
    
    
class SyntheticDataGenerator:
    """Generate synthetic repeat-sales transaction data"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Default market profiles by CBSA type
        self.market_profiles = {
            'high_growth': MarketProfile(
                base_price_mean=500000,
                base_price_std=200000,
                annual_appreciation_mean=0.06,
                annual_appreciation_std=0.02,
                volatility=0.15,
                transaction_rate=0.05
            ),
            'moderate_growth': MarketProfile(
                base_price_mean=300000,
                base_price_std=100000,
                annual_appreciation_mean=0.04,
                annual_appreciation_std=0.015,
                volatility=0.10,
                transaction_rate=0.04
            ),
            'low_growth': MarketProfile(
                base_price_mean=200000,
                base_price_std=50000,
                annual_appreciation_mean=0.02,
                annual_appreciation_std=0.01,
                volatility=0.08,
                transaction_rate=0.03
            )
        }
        
    def generate_cbsa_assignments(self, 
                                num_tracts: int,
                                num_cbsas: int) -> Dict[str, str]:
        """Assign tracts to CBSAs"""
        tract_ids = [f"{i:011d}" for i in range(num_tracts)]
        
        # Create CBSA IDs
        cbsa_ids = [f"{10000 + i * 20}" for i in range(num_cbsas)]
        
        # Assign tracts to CBSAs (varying sizes)
        assignments = {}
        remaining_tracts = tract_ids.copy()
        
        for i, cbsa_id in enumerate(cbsa_ids):
            # Last CBSA gets all remaining
            if i == len(cbsa_ids) - 1:
                cbsa_tracts = remaining_tracts
            else:
                # Random size between 5 and 50 tracts
                size = np.random.randint(5, min(50, len(remaining_tracts) - (num_cbsas - i - 1) * 5))
                cbsa_tracts = remaining_tracts[:size]
                remaining_tracts = remaining_tracts[size:]
            
            for tract_id in cbsa_tracts:
                assignments[tract_id] = cbsa_id
                
        return assignments
    
    def generate_property_base_data(self,
                                  num_properties: int,
                                  tract_assignments: Dict[str, str],
                                  cbsa_profiles: Dict[str, str]) -> pd.DataFrame:
        """Generate base property data"""
        properties = []
        
        tract_ids = list(tract_assignments.keys())
        
        for i in range(num_properties):
            tract_id = np.random.choice(tract_ids)
            cbsa_id = tract_assignments[tract_id]
            profile_type = cbsa_profiles.get(cbsa_id, 'moderate_growth')
            profile = self.market_profiles[profile_type]
            
            # Generate base price
            base_price = max(50000, np.random.normal(
                profile.base_price_mean,
                profile.base_price_std
            ))
            
            properties.append({
                'property_id': f"P{i:08d}",
                'tract_id': tract_id,
                'cbsa_id': cbsa_id,
                'base_price': base_price,
                'profile_type': profile_type
            })
            
        return pd.DataFrame(properties)
    
    def generate_market_indices(self,
                              start_year: int,
                              end_year: int,
                              cbsa_profiles: Dict[str, str]) -> pd.DataFrame:
        """Generate annual market appreciation indices by CBSA"""
        years = list(range(start_year, end_year + 1))
        
        indices = []
        for cbsa_id, profile_type in cbsa_profiles.items():
            profile = self.market_profiles[profile_type]
            
            # Generate correlated annual returns
            index_values = [1.0]  # Base year = 1.0
            
            for year in years[1:]:
                # Add some market cycles
                cycle_effect = 0.02 * np.sin(2 * np.pi * (year - start_year) / 10)
                
                # Add recession effects
                if year in [2001, 2008, 2009, 2020]:
                    annual_return = np.random.normal(-0.05, 0.03)
                else:
                    annual_return = np.random.normal(
                        profile.annual_appreciation_mean + cycle_effect,
                        profile.annual_appreciation_std
                    )
                
                new_index = index_values[-1] * (1 + annual_return)
                index_values.append(new_index)
            
            for i, year in enumerate(years):
                indices.append({
                    'cbsa_id': cbsa_id,
                    'year': year,
                    'index_value': index_values[i],
                    'annual_return': (index_values[i] / index_values[i-1] - 1) if i > 0 else 0
                })
                
        return pd.DataFrame(indices)
    
    def generate_transactions(self,
                            properties_df: pd.DataFrame,
                            market_indices_df: pd.DataFrame,
                            start_year: int,
                            end_year: int,
                            target_pairs: Optional[int] = None) -> pd.DataFrame:
        """Generate transaction history"""
        transactions = []
        
        # Calculate target based on transaction rates if not specified
        if target_pairs is None:
            avg_rate = np.mean([p.transaction_rate for p in self.market_profiles.values()])
            years = end_year - start_year + 1
            target_pairs = int(len(properties_df) * avg_rate * years * 0.3)  # 30% will be repeat sales
        
        # Track last sale for each property
        last_sales = {}
        
        # Generate initial sales
        for _, prop in properties_df.iterrows():
            profile = self.market_profiles[prop['profile_type']]
            
            # Random initial sale year
            first_year = np.random.randint(start_year, min(start_year + 10, end_year - 2))
            
            # Get market index
            idx_row = market_indices_df[
                (market_indices_df['cbsa_id'] == prop['cbsa_id']) &
                (market_indices_df['year'] == first_year)
            ].iloc[0]
            
            # Add property-specific variation
            property_factor = np.random.normal(1.0, profile.volatility)
            sale_price = prop['base_price'] * idx_row['index_value'] * property_factor
            
            sale_date = date(first_year, np.random.randint(1, 13), np.random.randint(1, 29))
            
            transaction = {
                'property_id': prop['property_id'],
                'tract_id': prop['tract_id'],
                'cbsa_id': prop['cbsa_id'],
                'sale_date': sale_date,
                'sale_price': max(10000, int(sale_price))
            }
            
            transactions.append(transaction)
            last_sales[prop['property_id']] = transaction
        
        # Generate repeat sales
        pairs_generated = 0
        attempts = 0
        max_attempts = target_pairs * 10
        
        while pairs_generated < target_pairs and attempts < max_attempts:
            attempts += 1
            
            # Pick a property that has sold
            property_id = np.random.choice(list(last_sales.keys()))
            last_sale = last_sales[property_id]
            
            prop = properties_df[properties_df['property_id'] == property_id].iloc[0]
            profile = self.market_profiles[prop['profile_type']]
            
            # Determine next sale year
            years_held = np.random.exponential(1 / profile.transaction_rate)
            years_held = max(1, min(int(years_held), 30))  # Between 1 and 30 years
            
            next_year = last_sale['sale_date'].year + years_held
            if next_year > end_year:
                continue
                
            # Get market appreciation
            idx_current = market_indices_df[
                (market_indices_df['cbsa_id'] == prop['cbsa_id']) &
                (market_indices_df['year'] == last_sale['sale_date'].year)
            ].iloc[0]
            
            idx_next = market_indices_df[
                (market_indices_df['cbsa_id'] == prop['cbsa_id']) &
                (market_indices_df['year'] == next_year)
            ].iloc[0]
            
            market_appreciation = idx_next['index_value'] / idx_current['index_value']
            
            # Add property-specific variation
            property_variation = np.random.normal(1.0, profile.volatility / np.sqrt(years_held))
            total_appreciation = market_appreciation * property_variation
            
            # Apply some realistic bounds
            total_appreciation = np.clip(total_appreciation, 0.5, 3.0)
            
            sale_date = date(next_year, np.random.randint(1, 13), np.random.randint(1, 29))
            sale_price = last_sale['sale_price'] * total_appreciation
            
            transaction = {
                'property_id': prop['property_id'],
                'tract_id': prop['tract_id'],
                'cbsa_id': prop['cbsa_id'],
                'sale_date': sale_date,
                'sale_price': max(10000, int(sale_price))
            }
            
            transactions.append(transaction)
            last_sales[property_id] = transaction
            pairs_generated += 1
        
        return pd.DataFrame(transactions)
    
    def create_repeat_sales_pairs(self, 
                                transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Convert transactions to repeat-sales pairs"""
        # Sort by property and date
        sorted_trans = transactions_df.sort_values(['property_id', 'sale_date'])
        
        pairs = []
        for property_id, group in sorted_trans.groupby('property_id'):
            if len(group) < 2:
                continue
                
            sales = group.to_dict('records')
            
            # Create pairs from consecutive sales
            for i in range(len(sales) - 1):
                pair = {
                    'property_id': property_id,
                    'tract_id': sales[i]['tract_id'],
                    'cbsa_id': sales[i]['cbsa_id'],
                    'first_sale_date': sales[i]['sale_date'],
                    'first_sale_price': sales[i]['sale_price'],
                    'second_sale_date': sales[i + 1]['sale_date'],
                    'second_sale_price': sales[i + 1]['sale_price']
                }
                pairs.append(pair)
                
        return pd.DataFrame(pairs)
    
    def generate_complete_dataset(self,
                                start_year: int = 1975,
                                end_year: int = 2023,
                                num_cbsas: int = 20,
                                num_tracts: int = 1000,
                                num_properties: int = 50000,
                                target_pairs: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate complete synthetic dataset
        
        Returns dict with:
        - 'transactions': All property transactions
        - 'pairs': Repeat-sales pairs
        - 'properties': Property base data
        - 'tracts': Tract-CBSA mapping
        - 'market_indices': Annual market indices by CBSA
        """
        print(f"Generating synthetic data from {start_year} to {end_year}...")
        
        # Assign tracts to CBSAs
        tract_assignments = self.generate_cbsa_assignments(num_tracts, num_cbsas)
        
        # Assign market profiles to CBSAs
        cbsa_list = list(set(tract_assignments.values()))
        cbsa_profiles = {}
        profile_types = list(self.market_profiles.keys())
        
        for i, cbsa_id in enumerate(cbsa_list):
            # Distribute profiles
            cbsa_profiles[cbsa_id] = profile_types[i % len(profile_types)]
        
        # Generate property base data
        print("Generating property base data...")
        properties_df = self.generate_property_base_data(
            num_properties, tract_assignments, cbsa_profiles
        )
        
        # Generate market indices
        print("Generating market indices...")
        market_indices_df = self.generate_market_indices(
            start_year, end_year, cbsa_profiles
        )
        
        # Generate transactions
        print("Generating transactions...")
        transactions_df = self.generate_transactions(
            properties_df, market_indices_df, start_year, end_year, target_pairs
        )
        
        # Create repeat-sales pairs
        print("Creating repeat-sales pairs...")
        pairs_df = self.create_repeat_sales_pairs(transactions_df)
        
        # Create tract summary
        tract_data = []
        for tract_id, cbsa_id in tract_assignments.items():
            tract_data.append({
                'tract_id': tract_id,
                'cbsa_id': cbsa_id,
                'state': tract_id[:2],
                'county': tract_id[2:5]
            })
        tracts_df = pd.DataFrame(tract_data)
        
        print(f"Generated {len(transactions_df):,} transactions")
        print(f"Generated {len(pairs_df):,} repeat-sales pairs")
        
        return {
            'transactions': transactions_df,
            'pairs': pairs_df,
            'properties': properties_df,
            'tracts': tracts_df,
            'market_indices': market_indices_df
        }
        
    def add_demographic_data(self,
                           tracts_df: pd.DataFrame,
                           years: List[int]) -> pd.DataFrame:
        """Add synthetic demographic data to tracts"""
        demographic_data = []
        
        for _, tract in tracts_df.iterrows():
            # Base values with some variation
            base_units = np.random.randint(100, 5000)
            base_value = np.random.normal(250000, 100000)
            base_college = np.random.beta(2, 5)  # Skewed towards lower values
            base_non_white = np.random.beta(2, 3)
            
            for year in years:
                # Add trends over time
                year_factor = (year - min(years)) / (max(years) - min(years))
                
                demographic_data.append({
                    'tract_id': tract['tract_id'],
                    'year': year,
                    'housing_units': int(base_units * (1 + 0.3 * year_factor)),
                    'median_value': max(50000, base_value * (1 + 0.5 * year_factor)),
                    'college_share': min(1.0, base_college * (1 + 0.2 * year_factor)),
                    'non_white_share': base_non_white,
                    'upb_total': base_units * base_value * 0.7 * np.random.uniform(0.5, 1.0)
                })
                
        return pd.DataFrame(demographic_data)