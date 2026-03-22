#!/usr/bin/env python3
"""
Machine Learning Training Pipeline for Performance Monitor System.
Trains models for time-series forecasting, anomaly detection, and disk failure prediction.
"""

import argparse
import logging
import yaml
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """Main ML training pipeline for performance monitoring."""
    
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
        
        # Model storage paths
        self.models_dir = Path(self.config['models']['storage_path'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.data_dir = Path(self.config['data']['local_storage'])
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Models dictionary
        self.models = {}
        
    def fetch_metrics_data(self, host='localhost', lookback_hours=24):
        """Fetch metrics data from InfluxDB for training."""
        logger.info(f"Fetching metrics data for host {host}, lookback {lookback_hours}h")
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Query for CPU metrics
        cpu_query = f'''
        from(bucket: "{self.config['data']['database']}")
          |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "cpu")
          |> filter(fn: (r) => r.host == "{host}")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "usage_user", "usage_system", "usage_idle", "usage_iowait"])
        '''
        
        # Query for memory metrics
        mem_query = f'''
        from(bucket: "{self.config['data']['database']}")
          |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "mem")
          |> filter(fn: (r) => r.host == "{host}")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "used_percent", "available_percent"])
        '''
        
        # Query for disk metrics
        disk_query = f'''
        from(bucket: "{self.config['data']['database']}")
          |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
          |> filter(fn: (r) => r._measurement == "disk")
          |> filter(fn: (r) => r.host == "{host}")
          |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "used_percent", "inodes_used"])
        '''
        
        try:
            # Execute queries
            cpu_df = self.query_api.query_data_frame(cpu_query)
            mem_df = self.query_api.query_data_frame(mem_query)
            disk_df = self.query_api.query_data_frame(disk_query)
            
            # Process DataFrames
            dataframes = []
            for df, name in [(cpu_df, 'cpu'), (mem_df, 'mem'), (disk_df, 'disk')]:
                if df is not None and len(df) > 0:
                    df['_time'] = pd.to_datetime(df['_time'])
                    df.set_index('_time', inplace=True)
                    df.columns = [f"{name}_{col}" for col in df.columns]
                    dataframes.append(df)
            
            # Merge all metrics
            if dataframes:
                merged_df = pd.concat(dataframes, axis=1)
                merged_df = merged_df.resample('1T').mean().ffill().bfill()
                logger.info(f"Fetched {len(merged_df)} data points with {merged_df.shape[1]} features")
                return merged_df
            else:
                logger.warning("No data fetched from InfluxDB")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching metrics data: {e}")
            return pd.DataFrame()
    
    def prepare_time_series_data(self, df, sequence_length=60):
        """Prepare time series data for LSTM training."""
        if len(df) < sequence_length * 2:
            logger.warning(f"Insufficient data for sequence length {sequence_length}")
            return None, None
        
        # Normalize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i+sequence_length])
            y.append(scaled_data[i+sequence_length, 0])  # Predict first feature (CPU usage)
        
        X = np.array(X)
        y = np.array(y)
        
        # Save scaler for later use
        self.scaler = scaler
        
        logger.info(f"Prepared time series data: X shape {X.shape}, y shape {y.shape}")
        return X, y
    
    def train_lstm_model(self, X_train, y_train, X_val, y_val):
        """Train LSTM model for time series forecasting."""
        logger.info("Training LSTM model...")
        
        model_config = self.config['models']['lstm']
        
        model = Sequential([
            LSTM(model_config['hidden_units'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(model_config['hidden_units'] // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=self.models_dir / 'lstm_best_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=model_config['epochs'],
            batch_size=model_config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['lstm'] = model
        logger.info("LSTM training completed")
        return history
    
    def train_anomaly_detection_model(self, X_train):
        """Train Isolation Forest model for anomaly detection."""
        logger.info("Training anomaly detection model...")
        
        model_config = self.config['anomaly']
        
        # Flatten time series data for anomaly detection
        if len(X_train.shape) == 3:
            X_flat = X_train.reshape(X_train.shape[0], -1)
        else:
            X_flat = X_train
        
        model = IsolationForest(
            contamination=model_config['contamination'],
            random_state=42,
            n_estimators=100
        )
        
        model.fit(X_flat)
        self.models['anomaly'] = model
        
        # Calculate anomaly scores
        scores = model.decision_function(X_flat)
        self.anomaly_threshold = np.percentile(scores, model_config['threshold_percentile'])
        
        logger.info(f"Anomaly detection trained. Threshold: {self.anomaly_threshold:.4f}")
        return model
    
    def train_disk_failure_model(self, smart_data):
        """Train model for disk failure prediction using SMART attributes."""
        logger.info("Training disk failure prediction model...")
        
        # This would normally use real SMART data with failure labels
        # For now, create synthetic data for demonstration
        if len(smart_data) == 0:
            logger.warning("No SMART data available for disk failure training")
            return None
        
        # Example features: temperature, power_on_hours, reallocated_sectors, etc.
        # In reality, you'd need labeled failure data
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        
        # For demonstration, create synthetic labels
        X = smart_data.values
        n_samples = len(X)
        y = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])  # 5% failure rate
        
        if np.sum(y) == 0:
            logger.warning("No positive failure samples in training data")
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model.fit(X_train, y_train)
        self.models['disk_failure'] = model
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        logger.info(f"Disk failure model AUC: {roc_auc_score(y_test, y_prob):.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
        return model
    
    def train_prophet_model(self, df, target_column):
        """Train Facebook Prophet model for seasonal forecasting."""
        logger.info(f"Training Prophet model for {target_column}...")
        
        # Prepare data for Prophet
        prophet_df = df[[target_column]].reset_index()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        model.fit(prophet_df)
        self.models['prophet'] = model
        
        # Make future predictions
        future = model.make_future_dataframe(periods=1440, freq='T')  # 1 day ahead
        forecast = model.predict(future)
        
        logger.info(f"Prophet model trained. Forecast shape: {forecast.shape}")
        return forecast
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models."""
        logger.info("Evaluating models...")
        
        results = {}
        
        # LSTM evaluation
        if 'lstm' in self.models:
            lstm_pred = self.models['lstm'].predict(X_test)
            mae = mean_absolute_error(y_test, lstm_pred)
            mse = mean_squared_error(y_test, lstm_pred)
            results['lstm'] = {'mae': mae, 'mse': mse}
            logger.info(f"LSTM MAE: {mae:.4f}, MSE: {mse:.4f}")
        
        # Anomaly detection evaluation
        if 'anomaly' in self.models:
            X_flat = X_test.reshape(X_test.shape[0], -1)
            scores = self.models['anomaly'].decision_function(X_flat)
            anomaly_rate = np.mean(scores < self.anomaly_threshold)
            results['anomaly'] = {'anomaly_rate': anomaly_rate}
            logger.info(f"Anomaly detection rate: {anomaly_rate:.4f}")
        
        return results
    
    def save_models(self):
        """Save trained models to disk."""
        logger.info("Saving models...")
        
        for name, model in self.models.items():
            if name == 'lstm':
                model.save(self.models_dir / f'{name}_model.h5')
            elif name == 'prophet':
                with open(self.models_dir / f'{name}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
            else:
                with open(self.models_dir / f'{name}_model.pkl', 'wb') as f:
                    pickle.dump(model, f)
        
        # Save scaler
        if hasattr(self, 'scaler'):
            with open(self.models_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        
        # Save anomaly threshold
        if hasattr(self, 'anomaly_threshold'):
            threshold_file = self.models_dir / 'anomaly_threshold.txt'
            with open(threshold_file, 'w') as f:
                f.write(str(self.anomaly_threshold))
        
        logger.info(f"Models saved to {self.models_dir}")
    
    def run_pipeline(self):
        """Execute the complete ML training pipeline."""
        logger.info("Starting ML training pipeline...")
        
        # 1. Fetch data
        df = self.fetch_metrics_data(
            host=self.config['data']['default_host'],
            lookback_hours=self.config['data']['lookback_hours']
        )
        
        if len(df) == 0:
            logger.error("No data available for training")
            return
        
        # 2. Prepare time series data
        X, y = self.prepare_time_series_data(df, sequence_length=self.config['models']['lstm']['sequence_length'])
        
        if X is None:
            logger.error("Insufficient data for time series preparation")
            return
        
        # 3. Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Further split for validation
        val_split_idx = int(len(X_train) * 0.8)
        X_train, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_train, y_val = y_train[:val_split_idx], y_train[val_split_idx:]
        
        # 4. Train models
        self.train_lstm_model(X_train, y_train, X_val, y_val)
        self.train_anomaly_detection_model(X_train)
        
        # 5. Train Prophet model for CPU usage
        self.train_prophet_model(df, target_column='cpu_usage_user')
        
        # 6. Evaluate models
        results = self.evaluate_models(X_test, y_test)
        
        # 7. Save models
        self.save_models()
        
        # 8. Log results to InfluxDB
        self.log_training_results(results)
        
        logger.info("ML training pipeline completed successfully")
    
    def log_training_results(self, results):
        """Log training results to InfluxDB for monitoring."""
        from influxdb_client import Point
        
        point = Point("ml_training") \
            .tag("host", "ml_pipeline") \
            .tag("pipeline", "training") \
            .field("status", "completed") \
            .field("timestamp", int(datetime.utcnow().timestamp()))
        
        for model_name, metrics in results.items():
            for metric_name, value in metrics.items():
                point.field(f"{model_name}_{metric_name}", float(value))
        
        try:
            self.write_api.write(
                bucket=self.config['data']['database'],
                org=self.config['data']['influxdb_org'],
                record=point
            )
            logger.info("Training results logged to InfluxDB")
        except Exception as e:
            logger.error(f"Error logging training results: {e}")

def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description='Train ML models for performance monitoring')
    parser.add_argument('--config', type=str, default='config/local.yaml',
                       help='Path to configuration file')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to fetch metrics from')
    parser.add_argument('--lookback', type=int, default=24,
                       help='Hours of historical data to fetch')
    
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return
    
    # Create and run pipeline
    pipeline = MLTrainingPipeline(args.config)
    
    # Update config with command line arguments
    pipeline.config['data']['default_host'] = args.host
    pipeline.config['data']['lookback_hours'] = args.lookback
    
    try:
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()