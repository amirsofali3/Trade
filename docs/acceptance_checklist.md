# RFE Pipeline Production-Safety Acceptance Checklist

This checklist guides manual smoke tests to verify the RFE pipeline improvements are working correctly.

## 1. Timestamp-Aligned Features and Labels ✅

### Test Steps:
1. Start the bot with `python main.py --config config.json`
2. Look for log message: **"Aligned features/labels on timestamps: N samples; groups=[...]"**
3. Verify N > 0 and groups include expected feature types

### Expected Behavior:
- ✅ No more "Length alignment: arrays=[20, 500, 500], common_len=20" with mismatched arrays
- ✅ Clear logging of aligned sample count
- ✅ Feature groups properly aligned on timestamps rather than truncated

### Log Examples:
```
Aligned features/labels on timestamps: 245 samples; groups=['ohlcv', 'indicator', 'sentiment', 'orderbook']
```

## 2. Adaptive, Leak-Safe Multiclass Labels ✅

### Test Steps:
1. Monitor initial label generation during RFE
2. Look for label adaptation sequence in logs
3. Verify no single-class labels in final output

### Expected Behavior:
- ✅ Initial threshold attempt: `Training labels: BUY=5, SELL=12, HOLD=483`
- ✅ If single-class: "Single-class detected, trying smaller threshold (0.001)"
- ✅ If still single-class: "Still single-class, using quantile-based adaptive thresholds"
- ✅ Final result: Multiple classes with reasonable distribution
- ✅ Lookahead configuration working (default 3 candles)

### Log Examples:
```
Training labels: BUY=12, SELL=15, HOLD=218 (post-alignment and cleaning)
```

## 3. RFE and Fallback System ✅

### Test Steps:
1. Watch RFE execution in logs
2. Test both sufficient data (>30 samples) and insufficient data (<30 samples) scenarios
3. Verify fallback mechanisms work

### Expected Behavior:
- ✅ Sufficient data: Normal RFE with RandomForest
- ✅ Insufficient data: **"RFE samples=X (<30 threshold) → using fallback=correlation|variance|mutual_info"**
- ✅ Fallback completes successfully with feature ranking
- ✅ Weight application: **"Applied RFE weights: strong=S, medium=M, weak=W"**

### Log Examples:
```
Applied RFE weights: strong=8, medium=5, weak=87
RFE feature selection completed!
```

## 4. Atomic, Thread-Safe Persistence ✅

### Test Steps:
1. Check for `rfe_results.json` and `version_history.json` files after RFE
2. Verify file integrity and contents
3. Test concurrent access scenarios

### Expected Behavior:
- ✅ Files created atomically (no partial writes)
- ✅ `rfe_results.json` contains `last_run` timestamp
- ✅ `version_history.json` contains `schema_version: 1`
- ✅ Thread locks prevent race conditions
- ✅ Temp files cleaned up on errors

### File Content Checks:
```json
// rfe_results.json
{
  "last_run": "2024-01-15T14:23:45.123456",
  "method": "RandomForestRFE",
  "n_features_selected": 15,
  "weights_summary": {"strong": 8, "medium": 5, "weak": 2}
}

// version_history.json  
{
  "schema_version": 1,
  "current_version": "1.0.0",
  "history": [...]
}
```

## 5. API Hardening ✅

### Test Steps:
1. Test all endpoints with `curl` or browser
2. Verify response structure and data types
3. Check error handling

### `/api/model-stats` Checklist:
- ✅ Returns 200 status code
- ✅ Contains numeric `training_accuracy` and `validation_accuracy`
- ✅ No mock timestamps like "2024-01-15T10:30:00"
- ✅ Real `model_version` string
- ✅ `performance_metrics` nested object exists
- ✅ Log message: **"/api/model-stats served successfully with model_version: X.X.X"**

### `/api/feature-selection` Checklist:
- ✅ `rfe_performed` field exists (true/false)
- ✅ `weights_summary` with counts: `{"strong": N, "medium": M, "weak": W}`
- ✅ `last_run` timestamp from actual RFE execution
- ✅ `method`, `n_features_selected`, `total_features` fields
- ✅ `selected_features` and `ranked_features` arrays

### `/api/version-history` Checklist:
- ✅ Safe JSON reading (handles FileNotFoundError, JSONDecodeError)
- ✅ Returns reasonable defaults if file missing
- ✅ Schema version field when available

### `/api/diagnostics` (New Endpoint):
- ✅ `buffer_length` (experience buffer size)
- ✅ `gating_weight_stats` (min/mean/max)
- ✅ `last_train_time` and `last_rfe_time` timestamps

## 6. Feature Dimension Projections ✅

### Test Steps:
1. Monitor neural network initialization logs
2. Look for projection layer creation messages
3. Verify no tensor size mismatch errors

### Expected Behavior:
- ✅ Log messages like: "Created projection for indicator: 100 → 128"
- ✅ All feature groups projected to `hidden_dim` consistently
- ✅ No "size of tensor a (128) vs b (100)" errors during training

## 7. Encoder NaN Handling ✅

### Test Steps:
1. Check for FutureWarning messages about deprecated pandas methods
2. Verify encoder transforms handle NaN values properly

### Expected Behavior:
- ✅ No `FutureWarning: The default value of numeric_only in DataFrame.fillna(method='ffill')` warnings
- ✅ Encoders use `df.ffill().bfill().fillna(0)` pattern
- ✅ Transform outputs contain no NaN/Inf values

## 8. Config Toggles and Data Volume ✅

### Test Steps:
1. Verify config.json contains new RFE settings
2. Test with different configuration values
3. Check data collection logs

### Config Keys to Verify:
```json
{
  "rfe": {
    "enabled": true,
    "lookahead": 3,
    "initial_threshold": 0.002,
    "min_samples": 30
  },
  "training": {
    "warmup_enabled": true
  },
  "online": {
    "update_interval": 60
  },
  "data": {
    "max_candles_for_rfe": 1000
  }
}
```

### Expected Behavior:
- ✅ RFE respects `rfe.enabled` flag
- ✅ Warmup respects `training.warmup_enabled` flag
- ✅ Data collection attempts up to `max_candles_for_rfe` samples
- ✅ Log message: **"Fetching up to 1000 candles for RFE training..."**

## 9. Enhanced Logging Messages ✅

### Key Phrases to Look For:
- ✅ **"Aligned features/labels on timestamps: N samples"**
- ✅ **"RFE samples=X (<30 threshold) → using fallback=..."**
- ✅ **"Applied RFE weights: strong=S, medium=M, weak=W"**
- ✅ **"Warmup training: batches=N loss: a -> b"**
- ✅ **"/api/model-stats served successfully"**

## 10. Demo Mode Safety ✅

### Test Steps:
1. Verify `demo_mode: true` in config.json
2. Check that no real trades are executed
3. Monitor trading logs for safety confirmations

### Expected Behavior:
- ✅ Config shows `"demo_mode": true` by default
- ✅ No real money trades executed
- ✅ All trading operations are simulated

## Manual Test Execution

### Quick Start:
1. Run: `python main.py --config config.json`
2. Wait for RFE to complete (~2-3 minutes)
3. Test API endpoints:
   ```bash
   curl http://localhost:5000/api/model-stats
   curl http://localhost:5000/api/feature-selection
   curl http://localhost:5000/api/version-history
   curl http://localhost:5000/api/diagnostics
   ```
4. Check log files for required phrases
5. Verify JSON files created: `rfe_results.json`, `version_history.json`

### Troubleshooting:
- If RFE fails: Check for sufficient historical data
- If APIs return errors: Verify bot instance is running
- If persistence fails: Check file permissions and disk space
- If tests fail: Run `python -m pytest test_*.py -v` for details

## Success Criteria:
- ✅ All log phrases present
- ✅ API endpoints return proper structure and data types
- ✅ No deprecated method warnings
- ✅ Files persist atomically with correct schema
- ✅ Feature alignment works with timestamps
- ✅ Label adaptation creates diverse classes
- ✅ RFE fallback mechanisms work
- ✅ Demo mode remains enabled by default