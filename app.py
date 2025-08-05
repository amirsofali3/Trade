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
        'portfolio': {'balance': 0, 'equity': 0, 'positions': []},
        'signals': [],
        'performance': {'daily': 0, 'weekly': 0, 'monthly': 0, 'all_time': 0},
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
    # New field to track active features
    'active_features': {
        'ohlcv': {'active': True, 'weight': 1.0},
        'indicators': {
            'rsi': {'active': True, 'weight': 0.8},
            'macd': {'active': True, 'weight': 0.7},
            'bollinger': {'active': True, 'weight': 0.9},
            # Add more indicators as needed
        },
        'sentiment': {'active': True, 'weight': 0.5},
        'orderbook': {'active': True, 'weight': 0.6},
        'tick_data': {'active': True, 'weight': 0.7}
    }
}

# Message queue for communication with the trading bot
bot_message_queue = queue.Queue()

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
    """Return the status of active features and their weights"""
    return jsonify(data_store['active_features'])

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

@app.route('/api/model-stats')
def model_stats():
    """Return model statistics"""
    return jsonify(data_store['model_stats'])

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