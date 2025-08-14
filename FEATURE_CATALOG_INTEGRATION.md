# Feature Catalog Integration for RFE Filtering

This implementation integrates `crypto_trading_feature_encyclopedia.csv` with the RFE (Recursive Feature Elimination) system to ensure proper feature filtering and must-keep preservation.

## Key Components

### 1. Feature Catalog (`models/feature_catalog.py`)
- **FeatureCatalog**: Main class that loads and parses the CSV file
- **Singleton Pattern**: Global `get_feature_catalog()` function for consistent access
- **Feature Sets**: 
  - `must_keep`: Features that should never be removed (Core OHLCV, marked indicators)
  - `rfe_pool`: Features eligible for RFE selection
  - `prereq`: Prerequisite features needed by other indicators
- **Fallback Behavior**: Graceful degradation when CSV is missing or corrupted

### 2. RFE Integration (`models/gating.py`)
- **Catalog Loading**: Automatically loads feature catalog during initialization
- **Feature Filtering**: Separates must-keep from RFE candidates before RFE processing
- **Result Combination**: Merges must-keep features (rank 0) with RFE selections (rank > 0)
- **Pool Statistics**: Tracks and logs candidate counts, must-keep counts, final selection counts
- **Edge Case Handling**: Supports rfe_n_features=0 and minimum feature scenarios

### 3. Active Feature Masks (`models/gating.py::build_active_feature_masks`)
- **Must-Keep Guarantee**: Always activates must-keep features even if not in RFE results
- **Core Group Preservation**: OHLCV group remains fully active
- **Prerequisite Inclusion**: Prerequisite features are preserved through `_is_core_or_prereq()`

### 4. API Metadata (`app.py`)
- **Enhanced Endpoints**: `/api/active-features` now includes:
  - `pool_candidates`: Number of features considered for RFE
  - `must_keep_count`: Number of features preserved as must-keep
  - `final_selected`: Total features selected (must-keep + RFE)

## Feature Classification Logic

### Core Features (Always Must-Keep)
- **OHLCV Data**: Open, High, Low, Close, Volume, Timestamp
- **Core Meta**: Symbol, exchange identifiers
- **Order Book L1**: Best bid/ask prices and sizes

### Must-Keep Indicators
- Features marked with `Must Keep (Not in RFE) = Yes` in CSV
- Examples: Some key moving averages, fundamental price ratios

### RFE Eligible Features
- Features marked with `RFE Eligible = Yes` in CSV
- Typically technical indicators, patterns, derived metrics
- Subject to RFE ranking and selection process

### Prerequisites
- Features marked with `Category = Prereq` in CSV
- Examples: True Range, Typical Price, Previous Close
- Required by other indicators, treated as must-keep through `_is_core_or_prereq()`

## Logging and Observability

The system provides comprehensive logging for monitoring RFE behavior:

```
RFE Pool: candidates=31, must_keep=5, selected_from_pool=8, final_selected=13
```

Where:
- `candidates`: Features eligible for RFE selection
- `must_keep`: Features preserved regardless of RFE
- `selected_from_pool`: Features actually selected by RFE
- `final_selected`: Total active features (must_keep + selected_from_pool)

## Backward Compatibility

The implementation maintains full backward compatibility:
- If CSV is missing: Falls back to basic OHLCV core features
- If features aren't in catalog: Treated as RFE-eligible by default
- If RFE fails: Must-keep features are still preserved
- Existing API responses remain unchanged, only enhanced with new metadata

## Configuration

The system respects existing configuration:
- `data.max_candles_for_rfe`: Maximum candles for RFE training (default: 1000)
- `rfe.enabled`: Enable/disable RFE processing
- `rfe.n_features`: Target number of features to select via RFE

## Testing

Comprehensive test suite covers:
1. **Catalog Loading**: CSV parsing, fallback behavior, feature classification
2. **RFE Exclusion**: Core features excluded from RFE candidates
3. **Must-Keep Preservation**: Features preserved even with zero RFE selection
4. **Active Mask Consistency**: Masks respect core feature requirements

## Usage Example

```python
from models.gating import FeatureGatingModule
from models.feature_catalog import get_feature_catalog

# Initialize with feature catalog integration
gating = FeatureGatingModule(feature_groups, rfe_n_features=8)

# Perform RFE with automatic filtering
results = gating.perform_rfe_selection(training_data, labels)

# Generate masks that respect must-keep requirements  
masks = gating.build_active_feature_masks()

# Access pool statistics
summary = gating.get_rfe_summary()
print(f"Candidates: {summary['pool_candidates']}, Must-keep: {summary['must_keep_count']}")
```

This integration ensures that critical market data features are never accidentally removed while still allowing RFE to optimize the selection of technical indicators and derived features.