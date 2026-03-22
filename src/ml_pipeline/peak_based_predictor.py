#!/usr/bin/env python3
"""
Peak-based anomaly predictor for Performance Monitor System.
Predicts anomaly likelihood based on historical peak values (max/min) over recent days.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PeakBasedPredictor:
    """
    Predicts anomaly likelihood based on historical peak values.
    
    Algorithm:
    1. Collect metrics for the last 'a' days (CPU, memory, disk, network)
    2. Calculate daily max/min peaks for each metric
    3. Establish baseline ranges from historical peaks
    4. Predict anomaly likelihood for today's time slots based on:
       - Distance from historical peaks
       - Time-of-day patterns
       - Rate of change
    """
    
    def __init__(self, config_path: str, lookback_days: int = 7):
        """
        Initialize predictor.
        
        Args:
            config_path: Path to configuration file
            lookback_days: Number of days to look back for historical data (default: 7)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.lookback_days = lookback_days
        self.time_slots = 24  # Predict for each hour of the day
        
        # InfluxDB client
        self.influx_client = influxdb_client.InfluxDBClient(
            url=self.config['data']['influxdb_host'],
            token=self.config['data']['influxdb_token'],
            org=self.config['data']['influxdb_org']
        )
        self.query_api = self.influx_client.query_api()
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Metrics to monitor
        self.metrics = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_rx', 'network_tx']
        
        # Historical peak data storage
        self.historical_peaks = {
            'max': {metric: [] for metric in self.metrics},
            'min': {metric: [] for metric in self.metrics},
            'hourly_patterns': {metric: np.zeros(24) for metric in self.metrics}
        }
        
        # Anomaly thresholds (configurable)
        self.thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 90.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'disk_usage': {'warning': 90.0, 'critical': 95.0},
            'network_rx': {'warning': 80.0, 'critical': 90.0},
            'network_tx': {'warning': 80.0, 'critical': 90.0}
        }
        
        logger.info(f"PeakBasedPredictor initialized with {lookback_days} days lookback")
    
    def fetch_historical_data(self, days: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical metrics data for specified number of days.
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            Dictionary of DataFrames keyed by metric name
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        data_frames = {}
        
        for metric in self.metrics:
            query = f'''
            from(bucket: "{self.config['data']['influxdb_bucket']}")
              |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
              |> filter(fn: (r) => r._measurement == "system_metrics")
              |> filter(fn: (r) => r._field == "{metric}")
              |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
            '''
            
            try:
                result = self.query_api.query(query)
                
                if result:
                    records = []
                    for table in result:
                        for record in table.records:
                            records.append({
                                'time': record.get_time(),
                                'value': record.get_value()
                            })
                    
                    if records:
                        df = pd.DataFrame(records)
                        df['time'] = pd.to_datetime(df['time'])
                        df.set_index('time', inplace=True)
                        data_frames[metric] = df
                        logger.info(f"Fetched {len(df)} records for {metric}")
                    else:
                        logger.warning(f"No data found for metric: {metric}")
                        data_frames[metric] = pd.DataFrame()
                else:
                    logger.warning(f"No query result for metric: {metric}")
                    data_frames[metric] = pd.DataFrame()
                    
            except Exception as e:
                logger.error(f"Error fetching data for {metric}: {e}")
                data_frames[metric] = pd.DataFrame()
        
        return data_frames
    
    def calculate_daily_peaks(self, data_frames: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate daily maximum and minimum peaks for each metric.
        
        Args:
            data_frames: Dictionary of metric DataFrames
            
        Returns:
            Dictionary with daily peaks for each metric
        """
        daily_peaks = {
            'max': {metric: [] for metric in self.metrics},
            'min': {metric: [] for metric in self.metrics},
            'dates': []
        }
        
        # Group data by day
        for metric, df in data_frames.items():
            if df.empty:
                continue
            
            # Resample to daily frequency
            daily_max = df.resample('D').max()
            daily_min = df.resample('D').min()
            
            for date in daily_max.index:
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in daily_peaks['dates']:
                    daily_peaks['dates'].append(date_str)
                
                daily_peaks['max'][metric].append(float(daily_max.loc[date, 'value']))
                daily_peaks['min'][metric].append(float(daily_min.loc[date, 'value']))
        
        return daily_peaks
    
    def analyze_hourly_patterns(self, data_frames: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Analyze hourly patterns for each metric.
        
        Args:
            data_frames: Dictionary of metric DataFrames
            
        Returns:
            Dictionary of hourly average patterns (24 values per metric)
        """
        hourly_patterns = {}
        
        for metric, df in data_frames.items():
            if df.empty:
                hourly_patterns[metric] = np.zeros(24)
                continue
            
            # Extract hour from timestamp
            df['hour'] = df.index.hour
            
            # Calculate average by hour
            hourly_avg = df.groupby('hour')['value'].mean()
            
            # Create full 24-hour array
            pattern = np.zeros(24)
            for hour, avg in hourly_avg.items():
                pattern[hour] = avg
            
            hourly_patterns[metric] = pattern
        
        return hourly_patterns
    
    def predict_anomaly_likelihood(self) -> pd.DataFrame:
        """
        Predict anomaly likelihood for today's time slots.
        
        Returns:
            DataFrame with anomaly predictions for each hour
        """
        logger.info("Predicting anomaly likelihood for today...")
        
        # Fetch historical data
        historical_data = self.fetch_historical_data(self.lookback_days)
        
        # Calculate historical peaks
        daily_peaks = self.calculate_daily_peaks(historical_data)
        
        # Analyze hourly patterns
        hourly_patterns = self.analyze_hourly_patterns(historical_data)
        
        # Store for future reference
        self.historical_peaks = daily_peaks
        self.historical_peaks['hourly_patterns'] = hourly_patterns
        
        # Prepare prediction results
        predictions = []
        
        # For each hour of today
        today = datetime.utcnow().date()
        
        for hour in range(self.time_slots):
            hour_predictions = {
                'timestamp': datetime.combine(today, datetime.min.time()) + timedelta(hours=hour),
                'hour': hour
            }
            
            # Calculate anomaly likelihood for each metric
            total_risk = 0
            metric_details = {}
            
            for metric in self.metrics:
                if metric in self.historical_peaks['max'] and self.historical_peaks['max'][metric]:
                    # Get historical max/min for this metric
                    historical_max = np.max(self.historical_peaks['max'][metric])
                    historical_min = np.min(self.historical_peaks['min'][metric])
                    
                    # Get expected value for this hour based on pattern
                    expected_value = hourly_patterns.get(metric, np.zeros(24))[hour]
                    
                    # Calculate risk factors
                    risk_factors = []
                    
                    # 1. Proximity to historical max
                    if historical_max > 0:
                        max_proximity = expected_value / historical_max if historical_max > 0 else 0
                        risk_factors.append(max_proximity * 0.5)
                    
                    # 2. Deviation from historical min
                    if historical_min > 0 and expected_value > 0:
                        min_deviation = (expected_value - historical_min) / historical_min if historical_min > 0 else 0
                        risk_factors.append(min(min_deviation, 1.0) * 0.3)
                    
                    # 3. Time-of-day pattern consistency
                    if len(self.historical_peaks['max'][metric]) >= 3:
                        # Check if this hour typically has high values
                        hour_rank = np.argsort(hourly_patterns.get(metric, np.zeros(24)))[::-1]
                        hour_risk = np.where(hour_rank == hour)[0][0] / 24.0
                        risk_factors.append(hour_risk * 0.2)
                    
                    # Calculate composite risk score (0-1)
                    if risk_factors:
                        metric_risk = np.mean(risk_factors)
                    else:
                        metric_risk = 0.0
                    
                    # Determine risk level
                    if metric_risk > 0.7:
                        risk_level = 'critical'
                    elif metric_risk > 0.5:
                        risk_level = 'warning'
                    else:
                        risk_level = 'normal'
                    
                    metric_details[metric] = {
                        'risk_score': round(metric_risk, 3),
                        'risk_level': risk_level,
                        'historical_max': round(historical_max, 2),
                        'historical_min': round(historical_min, 2),
                        'expected_value': round(expected_value, 2)
                    }
                    
                    total_risk += metric_risk
            
            # Calculate overall risk
            overall_risk = total_risk / len(self.metrics) if self.metrics else 0
            
            if overall_risk > 0.7:
                overall_level = 'critical'
            elif overall_risk > 0.5:
                overall_level = 'warning'
            else:
                overall_level = 'normal'
            
            hour_predictions['overall_risk_score'] = round(overall_risk, 3)
            hour_predictions['overall_risk_level'] = overall_level
            hour_predictions['metrics'] = metric_details
            
            predictions.append(hour_predictions)
        
        # Create DataFrame
        df_predictions = pd.DataFrame(predictions)
        
        # Sort by hour
        df_predictions.sort_values('hour', inplace=True)
        
        logger.info(f"Generated {len(df_predictions)} hourly predictions")
        
        return df_predictions
    
    def save_predictions_to_influx(self, predictions_df: pd.DataFrame):
        """
        Save prediction results to InfluxDB.
        
        Args:
            predictions_df: DataFrame with prediction results
        """
        records = []
        
        for _, row in predictions_df.iterrows():
            timestamp = row['timestamp']
            
            # Write overall prediction
            overall_record = {
                "measurement": "anomaly_predictions",
                "tags": {
                    "prediction_type": "peak_based",
                    "lookback_days": str(self.lookback_days)
                },
                "fields": {
                    "overall_risk_score": row['overall_risk_score'],
                    "risk_level_numeric": 2 if row['overall_risk_level'] == 'critical' else 
                                         1 if row['overall_risk_level'] == 'warning' else 0
                },
                "time": timestamp
            }
            records.append(overall_record)
            
            # Write metric-specific predictions
            for metric, details in row['metrics'].items():
                metric_record = {
                    "measurement": "metric_predictions",
                    "tags": {
                        "metric": metric,
                        "prediction_type": "peak_based"
                    },
                    "fields": {
                        "risk_score": details['risk_score'],
                        "risk_level_numeric": 2 if details['risk_level'] == 'critical' else 
                                             1 if details['risk_level'] == 'warning' else 0,
                        "historical_max": details['historical_max'],
                        "historical_min": details['historical_min'],
                        "expected_value": details['expected_value']
                    },
                    "time": timestamp
                }
                records.append(metric_record)
        
        # Write to InfluxDB
        if records:
            try:
                self.write_api.write(
                    bucket=self.config['data']['influxdb_bucket'],
                    org=self.config['data']['influxdb_org'],
                    record=records
                )
                logger.info(f"Saved {len(records)} prediction records to InfluxDB")
            except Exception as e:
                logger.error(f"Error saving predictions to InfluxDB: {e}")
    
    def generate_report(self, predictions_df: pd.DataFrame) -> str:
        """
        Generate human-readable report of predictions.
        
        Args:
            predictions_df: DataFrame with prediction results
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PEAK-BASED ANOMALY PREDICTION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Lookback period: {self.lookback_days} days")
        report_lines.append(f"Prediction date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append("")
        
        # Summary statistics
        critical_hours = predictions_df[predictions_df['overall_risk_level'] == 'critical']
        warning_hours = predictions_df[predictions_df['overall_risk_level'] == 'warning']
        
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total hours predicted: {len(predictions_df)}")
        report_lines.append(f"  Critical risk hours: {len(critical_hours)}")
        report_lines.append(f"  Warning risk hours: {len(warning_hours)}")
        report_lines.append("")
        
        # High-risk time slots
        high_risk = predictions_df[predictions_df['overall_risk_score'] >= 0.5]
        if not high_risk.empty:
            report_lines.append("HIGH-RISK TIME SLOTS (risk ≥ 0.5):")
            for _, row in high_risk.iterrows():
                hour_str = f"{row['hour']:02d}:00"
                report_lines.append(f"  {hour_str}: {row['overall_risk_level'].upper()} "
                                  f"(score: {row['overall_risk_score']:.3f})")
            report_lines.append("")
        
        # Most problematic metrics
        metric_risks = {}
        for _, row in predictions_df.iterrows():
            for metric, details in row['metrics'].items():
                if metric not in metric_risks:
                    metric_risks[metric] = []
                metric_risks[metric].append(details['risk_score'])
        
        if metric_risks:
            report_lines.append("METRIC RISK AVERAGES:")
            for metric, scores in metric_risks.items():
                avg_risk = np.mean(scores)
                max_risk = np.max(scores)
                report_lines.append(f"  {metric}: avg={avg_risk:.3f}, max={max_risk:.3f}")
        
        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        if len(critical_hours) > 0:
            report_lines.append("  1. Monitor system closely during critical hours")
            report_lines.append("  2. Consider scaling resources before peak times")
            report_lines.append("  3. Review historical patterns for optimization")
        elif len(warning_hours) > 0:
            report_lines.append("  1. Keep an eye on warning periods")
            report_lines.append("  2. Prepare contingency plans")
        else:
            report_lines.append("  1. System appears stable based on historical patterns")
            report_lines.append("  2. Continue regular monitoring")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Peak-based anomaly predictor')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--days', type=int, default=7, help='Lookback days for historical data')
    parser.add_argument('--output', choices=['console', 'influx', 'both'], default='both',
                       help='Output destination for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = PeakBasedPredictor(args.config, args.days)
    
    # Generate predictions
    predictions = predictor.predict_anomaly_likelihood()
    
    # Generate report
    report = predictor.generate_report(predictions)
    print(report)
    
    # Save