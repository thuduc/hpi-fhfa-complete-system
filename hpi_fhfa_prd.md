# House Price Index (HPI) - FHFA Product Requirements Document

## 1. Executive Summary

This PRD defines the requirements for implementing a flexible method of house price index construction using repeat-sales aggregates, based on the FHFA Working Paper 21-01 by Justin Contat and William D. Larson. The system constructs granular tract-level indices that can be aggregated using various weighting schemes to create customized city-level indices.

## 2. Background and Objectives

### 2.1 Purpose
- Develop a balanced panel of annual house price indices for single-family homes across Census tracts
- Address issues of within-city transaction composition and non-representative sampling
- Enable flexible construction of city-level indices tailored to specific use cases

### 2.2 Key Innovations
- Supertract approach for handling low transaction counts
- Multiple weighting schemes for different target indices
- Aggregation method that treats cities as baskets of submarkets

## 3. Technical Requirements

### 3.1 Core Equations

#### 3.1.1 Repeat-Sales Regression Model
```
pitτ = D'tτ δtτ + εitτ                                                    (1)
```
Where:
- `p`: (log) price for transacted housing unit i
- `D`: dummy variable capturing times of consecutive transactions
- `tτ`: subscripts indicate differencing between t and τ
- `ε`: IID error with mean zero and finite variance

#### 3.1.2 Pooled Index Estimation
```
p̂pooled(t,t-1) = δ̂t - δ̂t-1 ≡ δ̂t,t-1                                      (2)
```

#### 3.1.3 Aggregated Index with Sample Weights
```
E[p̂a,Sample(t,t-1)] = Σ(n=1 to N) wSample(n) E[δnt,t-1]                   (3)
```
Where:
- `wSample`: sample weights that sum to unity
- `N`: number of submarkets
- `δnt,t-1`: submarket appreciation rates

#### 3.1.4 General Aggregation Formula
```
p = Σ(n=1 to N) wn pn                                                     (4)
```

#### 3.1.5 Index Difference Formula
```
p - p' = Σ(wn - w'n)pn                                                    (5)
```

### 3.2 Algorithm Implementation

The system must implement the following algorithm for each period t = 1...T:

1. **Initialize**: Set Pa(t=0) = 1

2. **Construct Supertracts**: 
   - Create Nt supertracts for period t using period t and t-1 transaction counts
   - Minimum threshold: 40 half-pairs per year

3. **Calculate Submarket Indices**:
   - Estimate classic BMN price indices for each supertract using entire time sample
   - Capture δ̂n,t and δ̂n,t-1 for each supertract

4. **Apply Weights**:
   - Calculate or specify desired supertract weights wn

5. **Aggregate**:
   ```
   p̂a(t) = Σ(n=1 to Nt) wn(δ̂n,t - δ̂n,t-1)
   ```

6. **Construct Index Level**:
   ```
   P̂a(t) = P̂a(t-1) × exp(p̂a(t))
   ```

### 3.3 Supertract Construction Rules

- **Adjacency Rule**: If a tract lacks minimum observations, aggregate with nearest tract
- **Recursive Aggregation**: Continue aggregating until threshold is met
- **Centroid Distance**: Use tract centroid distances for determining nearest neighbors
- **Minimum Observations**: 40 half-pairs in each year of two-year window

### 3.4 Weighting Schemes

The system must support the following weight types:

1. **Sample Weights** (wSample)
   - Based on transaction counts/half-pairs
   - Approximates traditional pooled index

2. **Value Weights** (wValue)
   - Geometric Laspeyres formulation
   - Uses aggregate single-family housing value
   - Formula: snt = Vnt/ΣVnt where Vnt = pnt × qnt

3. **Unit Weights** (wUnit)
   - Based on share of single-family housing units
   - Represents all housing stock

4. **UPB Weights** (wUPB)
   - Based on aggregate outstanding loan balances
   - Enterprise (Fannie Mae/Freddie Mac) loans

5. **College Weights** (wCollege)
   - Based on college-educated population share
   - Static weights from 2010 Census

6. **Non-White Weights** (wNon-White)
   - Based on ACS share of non-white population
   - Static weights from 2010 Census

### 3.5 Data Requirements

#### 3.5.1 Transaction Data
- All purchase-money mortgages from:
  - Fannie Mae
  - Freddie Mac
  - Federal Housing Administration
  - CoreLogic deed recorder data

#### 3.5.2 Geographic Data
- Census tract definitions (2010)
- Tract centroids for distance calculations
- Core-based statistical area (CBSA) boundaries

#### 3.5.3 Demographic Data
- Decennial Census (1990, 2000, 2010)
- American Community Survey (2017)
- Housing unit counts
- Population characteristics

### 3.6 Quality Control Filters

Transaction pairs must pass the following filters:
1. Remove pairs transacting in same 12-month period
2. Remove pairs with compound annual growth rate > |30%|
3. Remove pairs with cumulative appreciation > 10x or < 0.75x prior sale

## 4. Functional Requirements

### 4.1 Index Calculation Module

**Function**: `calculate_repeat_sales_index(transactions, geography, time_period)`

**Inputs**:
- Transaction pairs with prices and dates
- Geographic identifiers (tract, CBSA)
- Time period specification

**Outputs**:
- Tract-level appreciation rates
- Supertract definitions
- Index values by period

### 4.2 Aggregation Module

**Function**: `aggregate_to_city_level(tract_indices, weight_type, cbsa)`

**Inputs**:
- Tract-level indices
- Weight type selection
- CBSA identifier

**Outputs**:
- City-level index
- Component weights
- Submarket contributions

### 4.3 Weight Calculation Module

**Function**: `calculate_weights(geography, weight_type, time_period)`

**Inputs**:
- Geographic unit
- Weight type specification
- Time period for dynamic weights

**Outputs**:
- Normalized weights summing to unity
- Weight components for validation

### 4.4 Supertract Generation Module

**Function**: `create_supertracts(tracts, transaction_counts, threshold)`

**Inputs**:
- Tract geometries and centroids
- Transaction counts by tract and period
- Minimum observation threshold

**Outputs**:
- Supertract definitions
- Tract-to-supertract mapping
- Validation statistics

## 5. Performance Requirements

### 5.1 Computational Efficiency
- Handle 63,122+ Census tracts
- Process 63+ million transaction pairs
- Estimate 1,037,465+ supertract-year regressions
- Support 47+ million parameters

### 5.2 Scalability
- Annual updates with new transaction data
- Extensible to additional weight types
- Support for multiple geographic aggregation levels

## 6. Validation Requirements

### 6.1 Index Validation
- Compare pooled index with sample-weighted aggregation index
- Verify weights sum to unity
- Check for index continuity across periods

### 6.2 Statistical Tests
- Verify OLS assumptions for repeat-sales regression
- Test for aggregation bias vs estimation error tradeoff
- Validate supertract homogeneity assumptions

## 7. Output Specifications

### 7.1 Index Outputs
- Annual frequency (1989-2021)
- Balanced panel across all geographies
- Base period normalization (index = 100 or 1)

### 7.2 Metadata Requirements
- Supertract definitions by period
- Transaction counts by geography
- Weight values by index type
- Quality flags for low-observation areas

## 8. Implementation Considerations

### 8.1 Computational Architecture
- Parallel processing for tract-level regressions
- Efficient sparse matrix operations for dummy variables
- Memory management for large transaction datasets

### 8.2 Update Frequency
- Annual index updates
- Quarterly capability for high-transaction areas
- Historical revision handling

### 8.3 Extension Points
- Additional weight types (climate risk, disaster exposure)
- Alternative geographic units (ZIP codes, school districts)
- Higher frequency indices (quarterly, monthly)

## 9. Mathematical Constants and Parameters

### 9.1 Default Parameters
- Minimum half-pairs threshold: 40
- Maximum CAGR filter: ±30%
- Maximum appreciation filter: 10x
- Minimum appreciation filter: 0.75x
- Base period index value: 1.0

### 9.2 Model Parameters
- Error variance: Assumed homoskedastic in base implementation
- Distance metric: Euclidean distance between tract centroids
- Aggregation hierarchy: Tract → Supertract → CBSA

## 10. Success Criteria

1. **Coverage**: Complete geographic coverage with no missing tracts
2. **Accuracy**: Correlation with existing indices where available
3. **Flexibility**: Support for all specified weight types
4. **Performance**: Annual updates completed within computational budget
5. **Reliability**: Stable indices without spurious volatility

## Appendix A: Variable Definitions

| Variable | Definition | Type |
|----------|------------|------|
| p | Log price | Continuous |
| δ | Period appreciation parameter | Continuous |
| w | Aggregation weight | [0,1] |
| N | Number of submarkets | Integer |
| n | Submarket index | Integer |
| t | Time period | Integer |
| D | Transaction timing dummy | Binary |
| ε | Regression error | Continuous |

## Appendix B: Weight Formula Specifications

### B.1 Value Weights (Laspeyres)
```
wValue(n,t) = V(n,t) / Σ(k=1 to N) V(k,t)
where V(n,t) = median_value(n,t) × housing_units(n,t)
```

### B.2 Sample Weights
```
wSample(n,t) = half_pairs(n,t) / Σ(k=1 to N) half_pairs(k,t)
```

### B.3 Unit Weights
```
wUnit(n,t) = housing_units(n,t) / Σ(k=1 to N) housing_units(k,t)
```