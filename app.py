import logging
import os
import json
from flask import Flask, render_template, jsonify, request
import pandas as pd
from datetime import datetime, timedelta
import threading
import queue
import time
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("WebInterface")

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Global data store
data_store = {
    'bot_status': 'offline',
    'last_update': None,
    'trading_data': {
        'portfolio': {'balance': 1000, 'equity': 1045.23, 'positions': []},
        'signals': [
            {
                'timestamp': datetime.now() - timedelta(minutes=15),
                'signal': 'BUY',
                'price': 42150.23,
                'confidence': 0.85
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=10),
                'signal': 'HOLD',
                'price': 42180.45,
                'confidence': 0.45
            },
            {
                'timestamp': datetime.now() - timedelta(minutes=5),
                'signal': 'SELL',
                'price': 42210.67,
                'confidence': 0.72
            }
        ],
        'performance': {'daily': 2.45, 'weekly': 5.23, 'monthly': 12.67, 'all_time': 15.34},
        'ohlcv': pd.DataFrame(),
        'indicators': {},
        'predictions': []
    },
    'system_stats': {
        'cpu_usage': 0,
        'memory_usage': 0,
        'uptime': 0,
        'errors': []
    },
    'settings': {
        'symbol': 'BTCUSDT',
        'timeframe': '5m',  # Changed to 5m as per requirements
        'risk_per_trade': 0.02,
        'take_profit': 0.05,
        'stop_loss': 0.03,
        'max_open_trades': 1
    },
    'model_stats': {
        'feature_importance': {},
        'update_counts': {},
        'market_regime': 'neutral'
    },
    # Enhanced field to track active features with more detailed data
    'active_features': {
        'ohlcv': {'active': True, 'weight': 0.85, 'last_value': 42150.23},
        'indicators': {
            'rsi': {'active': True, 'weight': 0.72, 'last_value': 58.45},
            'macd': {'active': True, 'weight': 0.68, 'last_value': 125.34},
            'bollinger_bands': {'active': True, 'weight': 0.81, 'last_value': 42200.15},
            'ema_20': {'active': True, 'weight': 0.65, 'last_value': 42075.89},
            'sma_50': {'active': True, 'weight': 0.45, 'last_value': 41950.32},
            'atr': {'active': True, 'weight': 0.58, 'last_value': 287.45},
            'stochastic': {'active': True, 'weight': 0.42, 'last_value': 62.18},
            'adx': {'active': True, 'weight': 0.55, 'last_value': 28.73},
            'volume_sma': {'active': True, 'weight': 0.38, 'last_value': 1245.67},
            'weak_indicator_1': {'active': False, 'weight': 0.01, 'last_value': 12.34},  # Weak feature
            'weak_indicator_2': {'active': False, 'weight': 0.008, 'last_value': 5.67},  # Very weak
        },
        'sentiment': {'active': True, 'weight': 0.52, 'last_value': 0.67},
        'orderbook': {'active': True, 'weight': 0.63, 'last_value': 2.45},
        'tick_data': {'active': True, 'weight': 0.71, 'last_value': 156.78}
    }
}

# Message queue for communication with the trading bot
bot_message_queue = queue.Queue()

# Global bot instance reference for accessing RFE and feature data
bot_instance = None

# Routes
@app.route('/')
def index():
    """Render the main dashboard page"""
    return render_template('index.html', 
                          bot_status=data_store['bot_status'],
                          last_update=data_store['last_update'],
                          settings=data_store['settings'])

@app.route('/api/status')
def status():
    """Return the current status of the trading bot"""
    return jsonify({
        'status': data_store['bot_status'],
        'last_update': data_store['last_update'],
        'system_stats': data_store['system_stats']
    })

@app.route('/api/portfolio')
def portfolio():
    """Return the current portfolio status"""
    return jsonify(data_store['trading_data']['portfolio'])

@app.route('/api/performance')
def performance():
    """Return performance metrics"""
    return jsonify(data_store['trading_data']['performance'])

@app.route('/api/signals')
def signals():
    """Return recent trading signals"""
    limit = request.args.get('limit', default=10, type=int)
    signals = data_store['trading_data']['signals'][-limit:]
    return jsonify(signals)

@app.route('/api/active-features')
def active_features():
    """Return the status of active features and their weights with enhanced data"""
    # Enhanced feature data with more detailed information
    enhanced_features = {}
    
    # Process each feature group from data_store
    for group_name, group_data in data_store['active_features'].items():
        if isinstance(group_data, dict):
            if group_name == 'indicators':
                # Process individual indicators
                enhanced_features[group_name] = {}
                for indicator_name, indicator_data in group_data.items():
                    weight = indicator_data.get('weight', 0.0)
                    is_active = weight > 0.01  # Features with weight <= 0.01 are considered inactive
                    
                    # Classify indicator strength
                    if weight <= 0.01:
                        status = 'weak'
                    elif weight < 0.05:
                        status = 'moderate'
                    else:
                        status = 'strong'
                    
                    enhanced_features[group_name][indicator_name] = {
                        'active': is_active,
                        'weight': weight,
                        'weight_percentage': weight * 100,
                        'status': status,
                        'last_value': group_data.get('last_value', 0.0),
                        'impact_score': weight * 10,  # Scale for display
                        'update_time': datetime.now().strftime('%H:%M:%S')
                    }
            else:
                # Process other feature groups
                weight = group_data.get('weight', 0.0)
                is_active = weight > 0.01
                
                # Classify feature strength
                if weight <= 0.01:
                    status = 'weak'
                elif weight < 0.05:
                    status = 'moderate'
                else:
                    status = 'strong'
                
                enhanced_features[group_name] = {
                    'active': is_active,
                    'weight': weight,
                    'weight_percentage': weight * 100,
                    'status': status,
                    'last_value': group_data.get('last_value', 0.0),
                    'impact_score': weight * 10,
                    'update_time': datetime.now().strftime('%H:%M:%S')
                }
    
    return jsonify(enhanced_features)

@app.route('/api/rfe-analysis')
def rfe_analysis():
    """Return detailed RFE analysis results"""
    if bot_instance and hasattr(bot_instance, 'gating'):
        gating = bot_instance.gating
        
        if hasattr(gating, 'rfe_selected_features') and gating.rfe_selected_features:
            # Organize by feature groups
            analysis_results = {
                'rfe_performed': getattr(gating, 'rfe_performed', False),
                'total_features_analyzed': len(gating.rfe_selected_features),
                'selected_features_count': len([f for f in gating.rfe_selected_features.values() if f['selected']]),
                'feature_breakdown': {},
                'top_features': [],
                'weak_features': [],
                'selection_summary': {},
                'feature_impact': {}
            }
            
            # Categorize features by group
            for feature_name, feature_data in gating.rfe_selected_features.items():
                group = feature_name.split('.')[0] if '.' in feature_name else 'other'
                
                if group not in analysis_results['feature_breakdown']:
                    analysis_results['feature_breakdown'][group] = {
                        'total': 0,
                        'selected': 0,
                        'features': []
                    }
                
                analysis_results['feature_breakdown'][group]['total'] += 1
                if feature_data['selected']:
                    analysis_results['feature_breakdown'][group]['selected'] += 1
                
                feature_info = {
                    'name': feature_name,
                    'selected': feature_data['selected'],
                    'rank': feature_data['rank'],
                    'importance': feature_data['importance'],
                    'weight': feature_data['importance'] if feature_data['selected'] else 0.01
                }
                
                analysis_results['feature_breakdown'][group]['features'].append(feature_info)
                
                # Top features (selected with high importance)
                if feature_data['selected'] and feature_data['importance'] > 0.1:
                    analysis_results['top_features'].append(feature_info)
                
                # Weak features (not selected or low importance)
                elif not feature_data['selected'] or feature_data['importance'] <= 0.05:
                    analysis_results['weak_features'].append(feature_info)
                
                # Feature impact summary
                analysis_results['feature_impact'][feature_name] = {
                    'impact_level': 'high' if feature_data['importance'] > 0.1 else 'medium' if feature_data['importance'] > 0.05 else 'low',
                    'weight_assigned': feature_data['importance'] if feature_data['selected'] else 0.01,
                    'selection_reason': 'RFE selected' if feature_data['selected'] else 'Weak feature - minimal weight'
                }
            
            # Sort top features by importance
            analysis_results['top_features'].sort(key=lambda x: x['importance'], reverse=True)
            analysis_results['weak_features'].sort(key=lambda x: x['importance'])
            
            # Selection summary
            analysis_results['selection_summary'] = {
                'methodology': 'Random Forest RFE' if len(gating.rfe_selected_features) > 50 else 'Correlation-based selection',
                'selection_criteria': 'Top features based on predictive importance',
                'weak_feature_handling': 'Assigned minimum weight of 0.01 to maintain tracking',
                'strong_feature_boost': 'High-impact features get proportional weights'
            }
            
            return jsonify(analysis_results)
    
    return jsonify({
        'rfe_performed': False,
        'message': 'RFE analysis not yet performed or no data available'
    })

@app.route('/api/feature-performance')
def feature_performance():
    """Return real-time feature performance metrics"""
    if bot_instance and hasattr(bot_instance, 'gating'):
        gating = bot_instance.gating
        
        if hasattr(gating, 'feature_performance'):
            performance_data = {
                'timestamp': datetime.now().isoformat(),
                'feature_groups': {},
                'adaptation_stats': {
                    'adaptation_rate': getattr(gating, 'adaptation_rate', 0.1),
                    'min_weight': getattr(gating, 'min_weight', 0.01),
                    'total_adaptations': 0
                }
            }
            
            for group_name, perf_data in gating.feature_performance.items():
                performance_data['feature_groups'][group_name] = {
                    'current_weight': perf_data.get('weight', 0.0),
                    'avg_performance': perf_data.get('avg_performance', 0.0),
                    'rfe_selected': perf_data.get('rfe_selected', False),
                    'rfe_rank': perf_data.get('rfe_rank', 999),
                    'adaptation_count': perf_data.get('adaptations', 0),
                    'status': 'strong' if perf_data.get('weight', 0) > 0.1 else 'moderate' if perf_data.get('weight', 0) > 0.05 else 'weak'
                }
                
                performance_data['adaptation_stats']['total_adaptations'] += perf_data.get('adaptations', 0)
            
            return jsonify(performance_data)
    
    return jsonify({
        'message': 'Feature performance data not available'
    })

@app.route('/api/chart-data')
def chart_data():
    """Return data for charts"""
    timeframe = request.args.get('timeframe', default='5m')  # Changed default to 5m
    limit = request.args.get('limit', default=100, type=int)
    
    ohlcv = data_store['trading_data']['ohlcv']
    if ohlcv.empty:
        return jsonify({'error': 'No data available'})
    
    # Limit the data
    ohlcv = ohlcv.tail(limit)
    
    # Convert to list format for plotting
    chart_data = {
        'timestamps': ohlcv.index.tolist() if isinstance(ohlcv.index, pd.DatetimeIndex) else list(range(len(ohlcv))),
        'open': ohlcv['open'].tolist(),
        'high': ohlcv['high'].tolist(),
        'low': ohlcv['low'].tolist(),
        'close': ohlcv['close'].tolist(),
        'volume': ohlcv['volume'].tolist()
    }
    
    # Add indicators if available
    indicators = data_store['trading_data']['indicators']
    for name, values in indicators.items():
        if len(values) >= len(ohlcv):
            chart_data[name] = values[-len(ohlcv):]
    
    # Add signals if available
    signals = data_store['trading_data']['signals']
    if signals:
        chart_data['signals'] = [s for s in signals if s['timestamp'] in ohlcv.index]
    
    return jsonify(chart_data)

@app.route('/api/learning-stats')
def learning_stats():
    """Return learning system statistics"""
    return jsonify(data_store.get('learning_stats', {
        'total_trades': 0,
        'success_rate': 0.0,
        'net_profit': 0.0,
        'model_updates': 0,
        'feature_performance': {}
    }))

@app.route('/api/confidence-validation')
def confidence_validation():
    """Return confidence validation data for signals"""
    # Mock confidence validation data - in real implementation this would track actual vs predicted outcomes
    recent_signals = data_store['trading_data']['signals'][-10:] if data_store['trading_data']['signals'] else []
    
    validation_data = {
        'total_signals': len(recent_signals),
        'validation_summary': {
            'high_confidence_correct': 0,
            'high_confidence_wrong': 0,
            'medium_confidence_correct': 0,
            'medium_confidence_wrong': 0,
            'low_confidence_correct': 0,
            'low_confidence_wrong': 0
        },
        'recent_validations': [],
        'overall_accuracy': 0.0,
        'confidence_bands': {
            'high': {'threshold': 0.7, 'accuracy': 0.85},
            'medium': {'threshold': 0.4, 'accuracy': 0.65},
            'low': {'threshold': 0.0, 'accuracy': 0.45}
        }
    }
    
    # Simulate recent signal validations
    for i, signal in enumerate(recent_signals):
        confidence = signal.get('confidence', 0.5)
        # Simulate actual outcome (in real system this would be based on trade results)
        actual_correct = (i % 3) != 0  # Simulate 66% accuracy for demo
        
        confidence_band = 'high' if confidence >= 0.7 else 'medium' if confidence >= 0.4 else 'low'
        
        validation_entry = {
            'timestamp': signal.get('timestamp', datetime.now().isoformat()),
            'signal': signal.get('signal', 'HOLD'),
            'predicted_confidence': confidence,
            'actual_correct': actual_correct,
            'confidence_band': confidence_band,
            'price': signal.get('price', 0.0)
        }
        
        validation_data['recent_validations'].append(validation_entry)
        
        # Update summary counts
        key = f"{confidence_band}_confidence_{'correct' if actual_correct else 'wrong'}"
        validation_data['validation_summary'][key] += 1
    
    # Calculate overall accuracy
    total_correct = sum(1 for v in validation_data['recent_validations'] if v['actual_correct'])
    if validation_data['recent_validations']:
        validation_data['overall_accuracy'] = total_correct / len(validation_data['recent_validations'])
    
    return jsonify(validation_data)

@app.route('/api/model-stats')
def model_stats():
    """Return enhanced model statistics with real data and proper error handling"""
    try:
        # Get base stats
        stats = data_store['model_stats'].copy()
        
        # Add detailed feature importance data
        stats['detailed_feature_importance'] = {}
        
        # Calculate feature importance from active features
        total_weight = 0
        for group_name, group_data in data_store['active_features'].items():
            if isinstance(group_data, dict) and 'weight' in group_data:
                total_weight += float(group_data['weight'])
            elif group_name == 'indicators':
                for indicator_name, indicator_data in group_data.items():
                    weight = indicator_data.get('weight', 0)
                    total_weight += float(weight) if isinstance(weight, (int, float)) else 0.0
        
        # Calculate percentages with type safety
        for group_name, group_data in data_store['active_features'].items():
            try:
                if isinstance(group_data, dict) and 'weight' in group_data:
                    weight = float(group_data['weight']) if isinstance(group_data['weight'], (int, float)) else 0.0
                    stats['detailed_feature_importance'][group_name] = {
                        'weight': weight,
                        'percentage': (weight / total_weight * 100) if total_weight > 0 else 0,
                        'status': 'active' if weight > 0.01 else 'inactive'
                    }
            except (TypeError, ValueError):
                # Handle invalid weight data
                stats['detailed_feature_importance'][group_name] = {
                    'weight': 0.0,
                    'percentage': 0.0,
                    'status': 'inactive'
                }
        
        # Get real performance metrics from learner
        performance_metrics = {}
        if bot_instance and hasattr(bot_instance, 'learner'):
            try:
                model_stats = bot_instance.learner.get_model_stats()
                performance_metrics = {
                    'training_accuracy': model_stats.get('training_accuracy', 0.0),
                    'validation_accuracy': model_stats.get('validation_accuracy', 0.0),
                    'model_version': model_stats.get('model_version', '1.0.0'),
                    'last_training_time': model_stats.get('last_update'),
                    'feature_count': len([g for g in data_store['active_features'].values() if isinstance(g, dict) and g.get('weight', 0) > 0.01])
                }
                
                # Add the performance metrics to main stats but also flatten to top level for compatibility
                stats.update(model_stats)
                
                # Try to read recent updates from version_history.json file
                try:
                    import json
                    with open('version_history.json', 'r', encoding='utf-8') as f:
                        version_data = json.load(f)
                        recent_updates = []
                        history = version_data.get('history', [])
                        
                        # Get last 3 updates  
                        for update in reversed(history[-3:]):
                            recent_updates.append({
                                'timestamp': update.get('timestamp'),
                                'type': update.get('update_type', 'unknown') + '_update',
                                'version_change': f"{update.get('from_version', 'unknown')} → {update.get('to_version', 'unknown')}"
                            })
                        
                        stats['recent_updates'] = recent_updates if recent_updates else [{'timestamp': None, 'type': 'no_updates', 'version_change': 'No version history available'}]
                        
                except (FileNotFoundError, json.JSONDecodeError):
                    # Fallback to learner's version history
                    if hasattr(bot_instance.learner, 'version_history'):
                        recent_updates = []
                        for update in reversed(bot_instance.learner.version_history[-3:]):
                            recent_updates.append({
                                'timestamp': update.get('timestamp'),
                                'type': update.get('update_type', 'unknown') + '_update', 
                                'version_change': f"{update.get('from_version', 'unknown')} → {update.get('to_version', 'unknown')}"
                            })
                        stats['recent_updates'] = recent_updates
                    else:
                        stats['recent_updates'] = [{'timestamp': None, 'type': 'no_updates', 'version_change': 'No updates available'}]
                
                logger.info(f"/api/model-stats served successfully with model_version: {model_stats.get('model_version', '1.0.0')}")
                
            except Exception as e:
                logger.error(f"Error getting model stats from learner: {str(e)}")
                # Set fallback values with proper types
                performance_metrics = {
                    'training_accuracy': 0.0,
                    'validation_accuracy': 0.0,
                    'model_version': '1.0.0',
                    'last_training_time': None,
                    'feature_count': 0
                }
                stats.update(performance_metrics)
                stats['recent_updates'] = [{'timestamp': None, 'type': 'error', 'version_change': f'Error: {str(e)[:50]}...'}]
        else:
            # Set fallback values when bot not available
            performance_metrics = {
                'training_accuracy': 0.0,
                'validation_accuracy': 0.0,
                'model_version': '1.0.0',
                'last_training_time': None,
                'feature_count': 0
            }
            stats.update(performance_metrics)
            stats['recent_updates'] = [{'timestamp': None, 'type': 'offline', 'version_change': 'Bot offline'}]
        
        # Add performance_metrics as nested object for backward compatibility
        stats['performance_metrics'] = performance_metrics
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Error in /api/model-stats: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'training_accuracy': 0.0,
            'validation_accuracy': 0.0,
            'model_version': '1.0.0',
            'recent_updates': [{'timestamp': None, 'type': 'error', 'version_change': 'Server error'}]
        }), 500

@app.route('/api/feature-selection')
def feature_selection():
    """Return detailed RFE feature selection results with all required fields"""
    try:
        # First try to read from external JSON file
        try:
            import json
            with open('rfe_results.json', 'r', encoding='utf-8') as f:
                rfe_data = json.load(f)
                
                # Ensure all required fields are present
                if 'weights_summary' not in rfe_data:
                    weights_mapping = rfe_data.get('weights_mapping', {})
                    rfe_data['weights_summary'] = {
                        'strong': weights_mapping.get('strong', 0),
                        'medium': weights_mapping.get('medium', 0), 
                        'weak': weights_mapping.get('weak', 0)
                    }
                
                # Ensure last_run field exists
                if 'last_run' not in rfe_data:
                    rfe_data['last_run'] = rfe_data.get('timestamp', datetime.now().isoformat())
                
                # Ensure rfe_performed field exists
                if 'rfe_performed' not in rfe_data:
                    rfe_data['rfe_performed'] = True
                
                return jsonify(rfe_data)
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.debug(f"Could not read rfe_results.json: {str(e)}")
        
        # Fallback to bot instance data
        if bot_instance and hasattr(bot_instance, 'gating'):
            gating = bot_instance.gating
            
            if gating.rfe_performed and gating.rfe_selected_features:
                selected_features = [k for k, v in gating.rfe_selected_features.items() if v['selected']]
                
                # Calculate weights summary
                weights_summary = {
                    'strong': len([f for f in gating.rfe_selected_features.values() if f['selected'] and f['importance'] >= 0.7]),
                    'medium': len([f for f in gating.rfe_selected_features.values() if f['selected'] and 0.3 <= f['importance'] < 0.7]),
                    'weak': len([f for f in gating.rfe_selected_features.values() if not f['selected'] or f['importance'] < 0.3])
                }
                
                result = {
                    'rfe_performed': True,
                    'method': 'RandomForestRFE' if hasattr(gating, 'rfe_model') and gating.rfe_model else 'fallback',
                    'last_run': datetime.now().isoformat(),
                    'n_features_selected': len(selected_features),
                    'total_features': len(gating.rfe_selected_features),
                    'selected_features': selected_features,
                    'ranked_features': [
                        {
                            'name': k,
                            'rank': v['rank'],
                            'importance': v['importance'],
                            'selected': v['selected']
                        }
                        for k, v in sorted(gating.rfe_selected_features.items(), 
                                         key=lambda x: x[1]['rank'])
                    ],
                    'weights_mapping': weights_summary,  # Keep for backward compatibility
                    'weights_summary': weights_summary   # New required field
                }
            else:
                result = {
                    'rfe_performed': False,
                    'method': 'none',
                    'last_run': None,
                    'n_features_selected': 0,
                    'total_features': 0,
                    'selected_features': [],
                    'ranked_features': [],
                    'weights_summary': {'strong': 0, 'medium': 0, 'weak': 0},
                    'message': 'RFE feature selection not yet performed'
                }
        else:
            result = {
                'rfe_performed': False,
                'method': 'none',
                'last_run': None,
                'n_features_selected': 0,
                'total_features': 0,
                'selected_features': [],
                'ranked_features': [],
                'weights_summary': {'strong': 0, 'medium': 0, 'weak': 0},
                'message': 'Bot instance not available'
            }
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in feature-selection endpoint: {str(e)}")
        return jsonify({
            'rfe_performed': False,
            'method': 'error',
            'last_run': None,
            'n_features_selected': 0,
            'total_features': 0,
            'selected_features': [],
            'ranked_features': [],
            'weights_summary': {'strong': 0, 'medium': 0, 'weak': 0},
            'error': str(e)
        }), 500

@app.route('/api/version-history')
def version_history():
    """Return chronological version history"""
    try:
        # Try to read from external file first
        try:
            import json
            with open('version_history.json', 'r', encoding='utf-8') as f:
                version_data = json.load(f)
                return jsonify(version_data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Fallback to bot instance
        if bot_instance and hasattr(bot_instance, 'learner') and hasattr(bot_instance.learner, 'version_history'):
            history = bot_instance.learner.version_history[-20:]  # Last 20 entries
            
            result = {
                'current_version': bot_instance.learner.model_version,
                'last_updated': datetime.now().isoformat(),
                'total_entries': len(history),
                'history': history
            }
        else:
            result = {
                'current_version': '1.0.0',
                'last_updated': None,
                'total_entries': 0,
                'history': []
            }
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in version-history endpoint: {str(e)}")
        return jsonify({
            'current_version': '1.0.0',
            'error': str(e)
        }), 500

@app.route('/api/diagnostics')  
def diagnostics():
    """Return diagnostic information for debugging"""
    try:
        result = {
            'buffer_length': 0,
            'gating_weight_stats': {'min': 0.0, 'mean': 0.0, 'max': 0.0},
            'last_train_time': None,
            'last_rfe_time': None,
            'system_info': {
                'timestamp': datetime.now().isoformat(),
                'bot_status': data_store.get('bot_status', 'unknown')
            }
        }
        
        if bot_instance:
            # Buffer length from learner
            if hasattr(bot_instance, 'learner') and hasattr(bot_instance.learner, 'experience_buffer'):
                buffer = bot_instance.learner.experience_buffer
                if isinstance(buffer, dict) and 'features' in buffer:
                    result['buffer_length'] = len(buffer['features'])
                
                # Last training time
                if hasattr(bot_instance.learner, 'last_update_time'):
                    result['last_train_time'] = bot_instance.learner.last_update_time.isoformat()
            
            # Gating weight statistics
            if hasattr(bot_instance, 'gating'):
                try:
                    weights = bot_instance.gating.get_rfe_weights()
                    if weights:
                        weight_values = [w for w in weights.values() if isinstance(w, (int, float))]
                        if weight_values:
                            result['gating_weight_stats'] = {
                                'min': round(min(weight_values), 4),
                                'mean': round(sum(weight_values) / len(weight_values), 4),
                                'max': round(max(weight_values), 4),
                                'count': len(weight_values)
                            }
                except Exception as e:
                    logger.debug(f"Error getting weight stats: {str(e)}")
            
            # Last RFE time from file
            try:
                import json
                with open('rfe_results.json', 'r', encoding='utf-8') as f:
                    rfe_data = json.load(f)
                    result['last_rfe_time'] = rfe_data.get('last_run') or rfe_data.get('timestamp')
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in diagnostics endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'buffer_length': 0,
            'gating_weight_stats': {'min': 0.0, 'mean': 0.0, 'max': 0.0},
            'last_train_time': None,
            'last_rfe_time': None
        }), 500

@app.route('/api/plot')
def plot():
    """Generate a plot for display in the web interface"""
    try:
        ohlcv = data_store['trading_data']['ohlcv']
        if ohlcv.empty:
            return jsonify({'error': 'No data available'})
        
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.02, row_heights=[0.7, 0.3],
                            subplot_titles=('Price', 'Volume'))
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=ohlcv.index,
            open=ohlcv['open'],
            high=ohlcv['high'],
            low=ohlcv['low'],
            close=ohlcv['close'],
            name='OHLCV'
        ), row=1, col=1)
        
        # Add volume trace
        fig.add_trace(go.Bar(
            x=ohlcv.index,
            y=ohlcv['volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.5)'
        ), row=2, col=1)
        
        # Add indicator traces if available
        indicators = data_store['trading_data']['indicators']
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
        color_idx = 0
        
        for name, values in indicators.items():
            if len(values) == len(ohlcv):
                fig.add_trace(go.Scatter(
                    x=ohlcv.index,
                    y=values,
                    mode='lines',
                    name=name,
                    line=dict(color=colors[color_idx % len(colors)])
                ), row=1, col=1)
                color_idx += 1
        
        # Add signals if available
        signals = data_store['trading_data']['signals']
        buy_x = []
        buy_y = []
        sell_x = []
        sell_y = []
        
        for signal in signals:
            timestamp = signal.get('timestamp')
            if timestamp in ohlcv.index:
                idx = ohlcv.index.get_loc(timestamp)
                price = signal.get('price', ohlcv['close'].iloc[idx])
                
                if signal['signal'] == 'BUY':
                    buy_x.append(timestamp)
                    buy_y.append(price)
                elif signal['signal'] == 'SELL':
                    sell_x.append(timestamp)
                    sell_y.append(price)
        
        if buy_x:
            fig.add_trace(go.Scatter(
                x=buy_x,
                y=buy_y,
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ), row=1, col=1)
        
        if sell_x:
            fig.add_trace(go.Scatter(
                x=sell_x,
                y=sell_y,
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ), row=1, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"{data_store['settings']['symbol']} - {data_store['settings']['timeframe']}",
            xaxis_title='Time',
            yaxis_title='Price',
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            margin=dict(l=50, r=50, b=50, t=80),
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='x'
        )
        
        # Convert to JSON
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graphJSON
        
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update bot settings"""
    if request.method == 'POST':
        try:
            new_settings = request.json
            
            # Validate and update settings
            if 'symbol' in new_settings:
                data_store['settings']['symbol'] = new_settings['symbol']
            if 'timeframe' in new_settings:
                data_store['settings']['timeframe'] = new_settings['timeframe']
            if 'risk_per_trade' in new_settings:
                risk = float(new_settings['risk_per_trade'])
                data_store['settings']['risk_per_trade'] = max(0.01, min(risk, 0.1))
            if 'take_profit' in new_settings:
                tp = float(new_settings['take_profit'])
                data_store['settings']['take_profit'] = max(0.01, tp)
            if 'stop_loss' in new_settings:
                sl = float(new_settings['stop_loss'])
                data_store['settings']['stop_loss'] = max(0.01, sl)
            if 'max_open_trades' in new_settings:
                data_store['settings']['max_open_trades'] = max(1, int(new_settings['max_open_trades']))
            
            # Send settings update to bot
            bot_message_queue.put({
                'type': 'settings_update',
                'settings': data_store['settings']
            })
            
            return jsonify({'status': 'success', 'settings': data_store['settings']})
            
        except Exception as e:
            logger.error(f"Error updating settings: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)})
    else:
        return jsonify(data_store['settings'])

@app.route('/api/control', methods=['POST'])
def control():
    """Control the trading bot (start/stop/restart)"""
    try:
        action = request.json.get('action')
        
        if action == 'start':
            if data_store['bot_status'] != 'running':
                # Send start command to bot
                bot_message_queue.put({'type': 'command', 'command': 'start'})
                data_store['bot_status'] = 'starting'
                return jsonify({'status': 'success', 'message': 'Starting bot...'})
            else:
                return jsonify({'status': 'warning', 'message': 'Bot is already running'})
                
        elif action == 'stop':
            if data_store['bot_status'] != 'stopped':
                # Send stop command to bot
                bot_message_queue.put({'type': 'command', 'command': 'stop'})
                data_store['bot_status'] = 'stopping'
                return jsonify({'status': 'success', 'message': 'Stopping bot...'})
            else:
                return jsonify({'status': 'warning', 'message': 'Bot is already stopped'})
                
        elif action == 'restart':
            # Send restart command to bot
            bot_message_queue.put({'type': 'command', 'command': 'restart'})
            data_store['bot_status'] = 'restarting'
            return jsonify({'status': 'success', 'message': 'Restarting bot...'})
            
        else:
            return jsonify({'status': 'error', 'message': 'Invalid action'})
            
    except Exception as e:
        logger.error(f"Error controlling bot: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

def update_from_trade(bot_data):
    """
    Update data store with information from trading bot
    
    Args:
        bot_data: Dictionary with updated trading bot data
    """
    try:
        # Update status
        if 'status' in bot_data:
            data_store['bot_status'] = bot_data['status']
        
        # Update last update time
        data_store['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Update trading data
        if 'portfolio' in bot_data:
            data_store['trading_data']['portfolio'] = bot_data['portfolio']
        
        if 'signals' in bot_data:
            data_store['trading_data']['signals'] = bot_data['signals']
        
        if 'performance' in bot_data:
            data_store['trading_data']['performance'] = bot_data['performance']
        
        if 'ohlcv' in bot_data:
            data_store['trading_data']['ohlcv'] = bot_data['ohlcv']
        
        if 'indicators' in bot_data:
            data_store['trading_data']['indicators'] = bot_data['indicators']
        
        if 'predictions' in bot_data:
            data_store['trading_data']['predictions'] = bot_data['predictions']
        
        # Update system stats
        if 'system_stats' in bot_data:
            data_store['system_stats'] = bot_data['system_stats']
        
        # Update model stats
        if 'model_stats' in bot_data:
            data_store['model_stats'] = bot_data['model_stats']
        
        # Update active features
        if 'active_features' in bot_data:
            data_store['active_features'] = bot_data['active_features']
        
        logger.debug("Updated data store from trading bot")
        
    except Exception as e:
        logger.error(f"Error updating from trading bot: {str(e)}")

def start_web_server(host='0.0.0.0', port=5000, debug=False):
    """
    Start the web server
    
    Args:
        host: Host to bind to
        port: Port to listen on
        debug: Enable debug mode
    """
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        logger.error(f"Error starting web server: {str(e)}")

# Run the app
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start web server
    start_web_server(debug=True)