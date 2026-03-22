#!/usr/bin/env python3
"""
Run advanced anomaly predictor with ensemble methods.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_pipeline.advanced_predictor import AdvancedPredictor
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def generate_mock_historical_data(days: int, metrics: list) -> dict:
    """
    Generate mock historical data for testing.
    
    Args:
        days: Number of days of data
        metrics: List of metric names
        
    Returns:
        Dictionary of DataFrames keyed by metric name
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    # Generate hourly data
    hours = days * 24
    timestamps = pd.date_range(start=start_time, end=end_time, periods=hours)
    
    data_frames = {}
    
    for metric in metrics:
        # Generate realistic-looking data with patterns
        base_values = {
            'cpu_usage': 45,
            'memory_usage': 60,
            'disk_usage': 70,
            'network_rx': 100,
            'network_tx': 80
        }
        
        base = base_values.get(metric, 50)
        
        # Add hourly pattern (higher during business hours)
        hourly_pattern = np.array([
            0.7, 0.6, 0.5, 0.5, 0.5, 0.6,  # 0-5
            0.8, 1.0, 1.2, 1.3, 1.3, 1.2,  # 6-11
            1.1, 1.3, 1.4, 1.3, 1.2, 1.1,  # 12-17
            1.0, 0.9, 0.8, 0.8, 0.7, 0.7   # 18-23
        ])
        
        # Generate data
        values = []
        for i, ts in enumerate(timestamps):
            hour = ts.hour
            # Base value + hourly pattern + noise
            val = base * hourly_pattern[hour] + np.random.normal(0, base * 0.1)
            values.append(max(0, val))  # Ensure non-negative
        
        df = pd.DataFrame({
            'time': timestamps,
            'value': values
        })
        df.set_index('time', inplace=True)
        data_frames[metric] = df
    
    return data_frames


def main():
    parser = argparse.ArgumentParser(description='Run advanced anomaly predictor')
    parser.add_argument('--config', default='config/local.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--days', type=int, default=7, 
                       help='Lookback days for historical data')
    parser.add_argument('--output', choices=['console', 'influx', 'both'], 
                       default='console', help='Output destination')
    parser.add_argument('--mock-data', action='store_true',
                       help='Use mock data for testing')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ADVANCED ANOMALY PREDICTOR")
    print("=" * 70)
    print(f"Lookback period: {args.days} days")
    print(f"Configuration: {args.config}")
    print(f"Output: {args.output}")
    print()
    
    # Initialize predictor
    try:
        predictor = AdvancedPredictor(args.config, args.days)
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        print("Using mock data for demonstration...")
        args.mock_data = True
    
    # Get historical data
    if args.mock_data:
        print("Generating mock historical data...")
        historical_data = generate_mock_historical_data(
            args.days, 
            predictor.metrics if 'predictor' in locals() else [
                'cpu_usage', 'memory_usage', 'disk_usage', 
                'network_rx', 'network_tx'
            ]
        )
        print(f"Generated data for {len(historical_data)} metrics")
        print()
        
        # Create predictor with mock config if needed
        if 'predictor' not in locals():
            predictor = AdvancedPredictor.__new__(AdvancedPredictor)
            predictor.lookback_days = args.days
            predictor.metrics = list(historical_data.keys())
            predictor.zscore_threshold = 2.5
            predictor.iqr_multiplier = 1.5
            predictor.ma_window = 7
            predictor.ma_std_multiplier = 2.0
            predictor.ewma_alpha = 0.3
            predictor.ensemble_weights = {
                'zscore': 0.25,
                'iqr': 0.20,
                'ma': 0.25,
                'ewma': 0.20,
                'pattern': 0.10
            }
            predictor.warning_threshold = 0.5
            predictor.critical_threshold = 0.7
    else:
        print("Fetching historical data from InfluxDB...")
        historical_data = predictor.fetch_historical_data(args.days)
    
    # Generate predictions
    print("\nGenerating predictions using ensemble methods...")
    print("Methods: Z-Score, IQR, Moving Average, EWMA, Time Pattern")
    print()
    
    predictions = predictor.predict_anomaly_likelihood(historical_data)
    
    # Generate report
    report = predictor.generate_report(predictions)
    print(report)
    
    # Save to InfluxDB if requested
    if args.output in ['influx', 'both'] and not args.mock_data:
        try:
            predictor.save_predictions_to_influx(predictions)
            print("\nPredictions saved to InfluxDB")
        except Exception as e:
            print(f"\nError saving to InfluxDB: {e}")
    
    # Save to CSV
    output_dir = Path("./reports/advanced_predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"predictions_{timestamp}.csv"
    
    # Flatten predictions for CSV
    csv_data = []
    for _, row in predictions.iterrows():
        for metric, details in row['metrics'].items():
            csv_data.append({
                'timestamp': row['timestamp'],
                'hour': row['hour'],
                'metric': metric,
                'risk_score': details['risk_score'],
                'risk_level': details['risk_level'],
                'trend': details['trend'],
                'trend_confidence': details['trend_confidence'],
                'expected_value': details['expected_value'],
                **{f"method_{k}": v for k, v in details['method_scores'].items()}
            })
    
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to: {csv_path}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
