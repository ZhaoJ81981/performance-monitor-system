#!/usr/bin/env python3
"""
Real‑time prediction and anomaly detection for Performance Monitor System.
Loads trained models and makes predictions on incoming metrics.
"""

import argparse
import logging
import yaml
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PredictionPipeline:
    """Real‑time prediction and anomaly detection pipeline."""
    
    def __init__(self, config_path):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # InfluxDB client
        self.influx_client = influxdb_client.InfluxDBClient(
            url=self.config['data']['influxdb_host'],
            token=self.config['data']['influxdb_token'],
            org=self.config['data']['influxdb_org']
        )
        self.query_api = self.influx_client.query_api()
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Model paths
        self.models_dir = Path(self.config['models']['storage_path'])
        
        # Load models
        self.models = {}
        self.load_models()
        
        # Load scaler
        self.scaler = None
        self.load_scaler()
        
        # Load anomaly threshold
        self.anomaly_threshold = None
        self.load_anomaly_threshold()
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_ttl = self.config['prediction'].get('cache_ttl', 300)
        
    def load_models(self):
        """Load trained models from disk."""
        logger.info("Loading trained models...")
        
        # LSTM model
        lstm_path = self.models_dir / 'lstm_model.h5'
        if lstm_path.exists():
            from tensorflow.keras.models import load_model
            self.models['lstm'] = load_model(lstm_path)
            logger.info(f"Loaded LSTM model from {lstm_path}")
        
        # Anomaly detection model
        anomaly_path = self.models_dir / 'anomaly_model.pkl'
        if anomaly_path.exists():
            with open(anomaly_path, 'rb') as f:
                self.models['anomaly'] = pickle.load(f)
            logger.info(f"Loaded anomaly model from {anomaly_path}")
        
        # Prophet model
        prophet_path = self.models_dir / 'prophet_model.pkl'
        if prophet_path.exists():
            with open(prophet_path, 'rb') as f:
                self.models['prophet'] = pickle.load(f)
            logger.info(f"Loaded Prophet model from {prophet_path}")
        
        # Disk failure model
        disk_path = self.models_dir / 'disk_failure_model.pkl'
        if disk_path.exists():
            with open(disk_path, 'rb') as f:
                self.models['disk_failure'] = pickle.load(f)
            logger.info(f"Loaded disk failure model from {disk_path}")
        
        if not self.models:
            logger.warning("No models loaded - predictions will be limited")
    
    def load_scaler(self):
        """Load data scaler."""
        scaler_path = self.models_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Loaded data scaler")
    
    def load_anomaly_threshold(self):
        """Load anomaly threshold."""
        threshold_path = self.models_dir / 'anomaly_threshold.txt'
        if threshold_path.exists():
            with open(threshold_path, 'r') as f:
                self.anomaly_threshold = float(f.read().strip())
            logger.info(f"Loaded anomaly threshold: {self.anomaly_threshold}")
    
    def fetch_recent_metrics(self, host='localhost', lookback_minutes=60):
        """Fetch recent metrics for prediction."""
        logger.info(f"Fetching recent metrics for {host}, lookback {lookback_minutes}m")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Query for multiple metrics
        query = f'''
        from(bucket: "{self.config['data']['database']}")
          |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
          |> filter(fn: (r) => r.host == "{host}")
          |> pivot(rowKey: ["_time"], columnKey: ["_measurement", "_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "cpu_usage_user", "mem_used_percent", "disk_used_percent"])
          |> fill(column: "cpu_usage_user", value: 0.0)
          |> fill(column: "mem_used_percent", value: 0.0)
          |> fill(column: "disk_used_percent", value: 0.0)
        '''
        
        try:
            df = self.query_api.query_data_frame(query)
            if df is not None and len(df) > 0:
                df['_time'] = pd.to_datetime(df['_time'])
                df.set_index('_time', inplace=True)
                df = df.resample('1T').mean().ffill().bfill()
                logger.info(f"Fetched {len(df)} data points")
                return df
            else:
                logger.warning("No recent metrics found")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            return pd.DataFrame()
    
    def predict_cpu_usage(self, metrics_df, forecast_horizon=60):
        """Predict CPU usage using LSTM model."""
        if 'lstm' not in self.models:
            logger.warning("LSTM model not available for CPU prediction")
            return None
        
        if len(metrics_df) < self.config['models']['lstm']['sequence_length']:
            logger.warning("Insufficient data for LSTM prediction")
            return None
        
        # Prepare data
        if self.scaler is None:
            logger.error("Scaler not loaded")
            return None
        
        # Use last sequence
        sequence_length = self.config['models']['lstm']['sequence_length']
        recent_data = metrics_df.tail(sequence_length)
        scaled_data = self.scaler.transform(recent_data)
        
        # Reshape for LSTM
        X = scaled_data.reshape(1, sequence_length, -1)
        
        # Make prediction
        prediction = self.models['lstm'].predict(X, verbose=0)
        
        # Inverse transform
        # Create dummy array for inverse transform
        dummy = np.zeros((1, scaled_data.shape[1]))
        dummy[0, 0] = prediction[0, 0]  # CPU usage is first feature
        prediction_original = self.scaler.inverse_transform(dummy)[0, 0]
        
        logger.info(f"CPU usage prediction: {prediction_original:.2f}%")
        return prediction_original
    
    def detect_anomalies(self, metrics_df):
        """Detect anomalies in recent metrics."""
        if 'anomaly' not in self.models:
            logger.warning("Anomaly model not available")
            return None
        
        if self.anomaly_threshold is None:
            logger.warning("Anomaly threshold not set")
            return None
        
        # Prepare data
        if len(metrics_df) < 10:
            logger.warning("Insufficient data for anomaly detection")
            return None
        
        # Use recent data
        recent_data = metrics_df.tail(10).values
        
        # Flatten for anomaly detection
        if len(recent_data.shape) == 3:
            recent_flat = recent_data.reshape(recent_data.shape[0], -1)
        else:
            recent_flat = recent_data
        
        # Calculate anomaly scores
        scores = self.models['anomaly'].decision_function(recent_flat)
        is_anomaly = scores < self.anomaly_threshold
        
        anomaly_count = np.sum(is_anomaly)
        anomaly_rate = anomaly_count / len(is_anomaly)
        
        logger.info(f"Anomaly detection: {anomaly_count} anomalies out of {len(is_anomaly)} ({anomaly_rate:.1%})")
        
        return {
            'scores': scores.tolist(),
            'is_anomaly': is_anomaly.tolist(),
            'anomaly_count': int(anomaly_count),
            'anomaly_rate': float(anomaly_rate),
            'threshold': float(self.anomaly_threshold)
        }
    
    def forecast_with_prophet(self, metric_name, periods=1440):
        """Generate forecast using Prophet model."""
        if 'prophet' not in self.models:
            logger.warning("Prophet model not available")
            return None
        
        # Create future dataframe
        future = self.models['prophet'].make_future_dataframe(periods=periods, freq='T')
        forecast = self.models['prophet'].predict(future)
        
        # Extract relevant columns
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        result.columns = ['timestamp', 'prediction', 'lower_bound', 'upper_bound']
        
        logger.info(f"Prophet forecast generated: {len(result)} periods")
        return result
    
    def predict_disk_failure(self, smart_data):
        """Predict disk failure probability using SMART attributes."""
        if 'disk_failure' not in self.models:
            logger.warning("Disk failure model not available")
            return None
        
        if smart_data is None or len(smart_data) == 0:
            logger.warning("No SMART data available")
            return None
        
        # Ensure data has required features
        required_features = self.config['models']['disk_failure']['smart_attributes']
        missing = [f for f in required_features if f not in smart_data.columns]
        if missing:
            logger.warning(f"Missing SMART features: {missing}")
            return None
        
        # Prepare data
        X = smart_data[required_features].values
        
        # Predict probability
        probability = self.models['disk_failure'].predict_proba(X)[:, 1]
        
        # Check against threshold
        threshold = self.config['models']['disk_failure']['failure_threshold']
        will_fail = probability > threshold
        
        logger.info(f"Disk failure prediction: max probability {probability.max():.4f}")
        
        return {
            'probabilities': probability.tolist(),
            'will_fail': will_fail.tolist(),
            'threshold': float(threshold),
            'max_probability': float(probability.max())
        }
    
    def log_predictions(self, host, predictions):
        """Log predictions to InfluxDB for monitoring."""
        from influxdb_client import Point
        
        timestamp = datetime.utcnow()
        
        points = []
        
        # CPU prediction
        if 'cpu_prediction' in predictions:
            point = Point("prediction") \
                .tag("host", host) \
                .tag("metric", "cpu_usage") \
                .tag("model", "lstm") \
                .field("value", float(predictions['cpu_prediction'])) \
                .time(timestamp)
            points.append(point)
        
        # Anomaly detection
        if 'anomaly' in predictions:
            point = Point("anomaly") \
                .tag("host", host) \
                .tag("metric", "composite") \
                .field("rate", float(predictions['anomaly']['anomaly_rate'])) \
                .field("count", int(predictions['anomaly']['anomaly_count'])) \
                .field("threshold", float(predictions['anomaly']['threshold'])) \
                .time(timestamp)
            points.append(point)
        
        # Disk failure
        if 'disk_failure' in predictions:
            point = Point("disk_failure") \
                .tag("host", host) \
                .tag("metric", "failure_probability") \
                .field("probability", float(predictions['disk_failure']['max_probability'])) \
                .field("above_threshold", float(predictions['disk_failure']['max_probability']) > 
                       float(predictions['disk_failure']['threshold'])) \
                .time(timestamp)
            points.append(point)
        
        # Write to InfluxDB
        if points:
            try:
                self.write_api.write(
                    bucket=self.config['data']['database'],
                    org=self.config['data']['influxdb_org'],
                    record=points
                )
                logger.info(f"Logged {len(points)} prediction points to InfluxDB")
            except Exception as e:
                logger.error(f"Error logging predictions: {e}")
    
    def run_prediction_cycle(self, host='localhost'):
        """Run one complete prediction cycle."""
        logger.info(f"Starting prediction cycle for host {host}")
        
        # Fetch recent metrics
        metrics_df = self.fetch_recent_metrics(host=host)
        
        if metrics_df.empty:
            logger.warning("No metrics available for prediction")
            return
        
        predictions = {}
        
        # 1. Predict CPU usage
        cpu_pred = self.predict_cpu_usage(metrics_df)
        if cpu_pred is not None:
            predictions['cpu_prediction'] = cpu_pred
        
        # 2. Detect anomalies
        anomaly_result = self.detect_anomalies(metrics_df)
        if anomaly_result is not None:
            predictions['anomaly'] = anomaly_result
        
        # 3. Generate Prophet forecast
        if 'prophet' in self.models:
            forecast = self.forecast_with_prophet('cpu_usage_user')
            if forecast is not None:
                predictions['prophet_forecast'] = forecast.to_dict('records')
        
        # 4. Predict disk failure (would need SMART data)
        # This is placeholder - would need actual SMART data
        
        # Log predictions
        self.log_predictions(host, predictions)
        
        # Check for alerts
        self.check_alerts(host, predictions)
        
        logger.info(f"Prediction cycle completed. Generated {len(predictions)} predictions")
        return predictions
    
    def check_alerts(self, host, predictions):
        """Check predictions against alert thresholds and trigger alerts."""
        alert_config = self.config.get('alerting', {})
        if not alert_config.get('enabled', False):
            return
        
        alerts = []
        
        # CPU usage alert
        if 'cpu_prediction' in predictions:
            cpu_threshold = alert_config.get('cpu_threshold', 90)
            if predictions['cpu_prediction'] > cpu_threshold:
                alerts.append({
                    'severity': 'warning',
                    'metric': 'cpu_usage',
                    'value': predictions['cpu_prediction'],
                    'threshold': cpu_threshold,
                    'message': f'Predicted CPU usage {predictions["cpu_prediction"]:.1f}% exceeds threshold {cpu_threshold}%'
                })
        
        # Anomaly alert
        if 'anomaly' in predictions:
            anomaly_threshold = alert_config.get('anomaly_rate_threshold', 0.3)
            if predictions['anomaly']['anomaly_rate'] > anomaly_threshold:
                alerts.append({
                    'severity': 'critical',
                    'metric': 'anomaly_rate',
                    'value': predictions['anomaly']['anomaly_rate'],
                    'threshold': anomaly_threshold,
                    'message': f'High anomaly rate {predictions["anomaly"]["anomaly_rate"]:.1%} exceeds threshold {anomaly_threshold:.1%}'
                })
        
        # Disk failure alert
        if 'disk_failure' in predictions:
            failure_threshold = predictions['disk_failure']['threshold']
            max_prob = predictions['disk_failure']['max_probability']
            if max_prob > failure_threshold:
                alerts.append({
                    'severity': 'critical',
                    'metric': 'disk_failure',
                    'value': max_prob,
                    'threshold': failure_threshold,
                    'message': f'Disk failure probability {max_prob:.1%} exceeds threshold {failure_threshold:.1%}'
                })
        
        # Send alerts
        if alerts:
            self.send_alerts(host, alerts)
    
    def send_alerts(self, host, alerts):
        """Send alerts via configured channels."""
        alert_config = self.config.get('alerting', {})
        
        for alert in alerts:
            # Log to InfluxDB
            from influxdb_client import Point
            
            point = Point("alert") \
                .tag("host", host) \
                .tag("severity", alert['severity']) \
                .tag("metric", alert['metric']) \
                .field("value", float(alert['value'])) \
                .field("threshold", float(alert['threshold'])) \
                .field("message", alert['message']) \
                .time(datetime.utcnow())
            
            try:
                self.write_api.write(
                    bucket=self.config['data']['database'],
                    org=self.config['data']['influxdb_org'],
                    record=point
                )
                logger.info(f"Alert logged: {alert['message']}")
            except Exception as e:
                logger.error(f"Error logging alert: {e}")
            
            # Additional alert channels could be added here (email, Slack, etc.)
            # Based on alert_config
    
    def run_continuous(self, interval_seconds=300):
        """Run prediction pipeline continuously at specified interval."""
        import time
        
        logger.info(f"Starting continuous prediction pipeline with {interval_seconds}s interval")
        
        try:
            while True:
                start_time = time.time()
                
                # Run prediction cycle for default host
                self.run_prediction_cycle()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(1, interval_seconds - elapsed)
                
                logger.info(f"Cycle completed in {elapsed:.1f}s. Sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Prediction pipeline stopped by user")
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {e}")
            raise

def main():
    """Main entry point for prediction script."""
    parser = argparse.ArgumentParser(description='Run predictions and anomaly detection')
    parser.add_argument('--config', type=str, default='config/local.yaml',
                       help='Path to configuration file')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to monitor')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously')
    parser.add_argument('--interval', type=int, default=300,
                       help='Interval in seconds for continuous mode')
    
    args = parser.parse_args()
    
    # Check config
    import os
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # Create and run pipeline
    pipeline = PredictionPipeline(args.config)
    
    if args.continuous:
        pipeline.run_continuous(interval_seconds=args.interval)
    else:
        predictions = pipeline.run_prediction_cycle(host=args.host)
        if predictions:
            logger.info(f"Predictions: {predictions.keys()}")
        else:
            logger.warning("No predictions generated")

if __name__ == "__main__":
    main()