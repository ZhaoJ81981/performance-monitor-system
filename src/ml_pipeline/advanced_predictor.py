#!/usr/bin/env python3
"""
Advanced anomaly predictor for Performance Monitor System.
Uses multiple statistical methods beyond simple max/min thresholds.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import yaml
from pathlib import Path
from scipy import stats
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedPredictor:
    """
    Advanced anomaly predictor using multiple statistical methods:
    
    1. Z-Score Analysis: Detect values deviating from mean
    2. IQR Method: Identify outliers using interquartile range
    3. Moving Average + Standard Deviation: Dynamic thresholds
    4. Trend Analysis: Exponential weighted moving average
    5. Time Series Decomposition: Separate trend, seasonality, residual
    6. Ensemble Method: Combine multiple approaches for robust prediction
    """
    
    def __init__(self, config_path: str, lookback_days: int = 30):
        """
        Initialize advanced predictor.
        
        Args:
            config_path: Path to configuration file
            lookback_days: Number of days to look back for historical data
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.lookback_days = lookback_days
        self.time_slots = 24
        
        # Get configuration parameters
        prediction_config = self.config.get('prediction', {}).get('advanced', {})
        
        # Z-Score parameters
        self.zscore_threshold = prediction_config.get('zscore_threshold', 2.5)
        
        # IQR parameters
        self.iqr_multiplier = prediction_config.get('iqr_multiplier', 1.5)
        
        # Moving average parameters
        self.ma_window = prediction_config.get('ma_window', 7)  # 7-hour window
        self.ma_std_multiplier = prediction_config.get('ma_std_multiplier', 2.0)
        
        # EWMA parameters
        self.ewma_alpha = prediction_config.get('ewma_alpha', 0.3)
        
        # Ensemble weights
        self.ensemble_weights = prediction_config.get('ensemble_weights', {
            'zscore': 0.25,
            'iqr': 0.20,
            'ma': 0.25,
            'ewma': 0.20,
            'pattern': 0.10
        })
        
        # Metrics to monitor
        self.metrics = self.config.get('prediction', {}).get('peak_based', {}).get(
            'metrics_to_monitor',
            ['cpu_usage', 'memory_usage', 'disk_usage', 'network_rx', 'network_tx']
        )
        
        # Risk thresholds
        risk_config = self.config.get('prediction', {}).get('peak_based', {}).get('risk_thresholds', {})
        self.warning_threshold = risk_config.get('warning', 0.5)
        self.critical_threshold = risk_config.get('critical', 0.7)
        
        # Historical data storage
        self.historical_data = {}
        
        logger.info(f"AdvancedPredictor initialized with {lookback_days} days lookback")
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
    
    def zscore_anomaly_score(self, values: np.ndarray, current: float) -> float:
        """
        Calculate anomaly score using Z-score method.
        
        Args:
            values: Historical values
            current: Current value to check
            
        Returns:
            Anomaly score (0-1)
        """
        if len(values) < 3:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return 0.0
        
        z_score = abs((current - mean) / std)
        
        # Convert z-score to anomaly score (0-1)
        # z_score_threshold corresponds to score of 0.5
        score = min(z_score / (self.zscore_threshold * 2), 1.0)
        
        return score
    
    def iqr_anomaly_score(self, values: np.ndarray, current: float) -> float:
        """
        Calculate anomaly score using IQR method.
        
        Args:
            values: Historical values
            current: Current value to check
            
        Returns:
            Anomaly score (0-1)
        """
        if len(values) < 4:
            return 0.0
        
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return 0.0
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        # Calculate distance from bounds
        if current < lower_bound:
            distance = (lower_bound - current) / iqr
        elif current > upper_bound:
            distance = (current - upper_bound) / iqr
        else:
            return 0.0
        
        # Convert to score (0-1)
        score = min(distance / self.iqr_multiplier, 1.0)
        
        return score
    
    def moving_average_score(self, values: np.ndarray, current: float) -> float:
        """
        Calculate anomaly score using moving average + std deviation.
        
        Args:
            values: Historical values
            current: Current value to check
            
        Returns:
            Anomaly score (0-1)
        """
        if len(values) < self.ma_window:
            return 0.0
        
        # Use last N values for moving average
        recent_values = values[-self.ma_window:]
        ma = np.mean(recent_values)
        std = np.std(recent_values)
        
        if std == 0:
            return 0.0
        
        # Calculate deviation from moving average
        deviation = abs(current - ma) / std
        
        # Convert to score (0-1)
        score = min(deviation / (self.ma_std_multiplier * 2), 1.0)
        
        return score
    
    def ewma_anomaly_score(self, values: np.ndarray, current: float) -> float:
        """
        Calculate anomaly score using Exponentially Weighted Moving Average.
        
        Args:
            values: Historical values
            current: Current value to check
            
        Returns:
            Anomaly score (0-1)
        """
        if len(values) < 3:
            return 0.0
        
        # Calculate EWMA
        ewma = values[0]
        for val in values[1:]:
            ewma = self.ewma_alpha * val + (1 - self.ewma_alpha) * ewma
        
        # Calculate EWMA of squared deviations for volatility
        ewma_var = np.var(values[:10]) if len(values) >= 10 else np.var(values)
        for val in values:
            deviation = val - ewma
            ewma_var = self.ewma_alpha * deviation**2 + (1 - self.ewma_alpha) * ewma_var
        
        ewma_std = np.sqrt(ewma_var)
        
        if ewma_std == 0:
            return 0.0
        
        # Calculate deviation from EWMA
        deviation = abs(current - ewma) / ewma_std
        
        # Convert to score (0-1)
        score = min(deviation / 3.0, 1.0)
        
        return score
    
    def time_pattern_score(self, hourly_values: Dict[int, List[float]], 
                          hour: int, current: float) -> float:
        """
        Calculate anomaly score based on time-of-day pattern.
        
        Args:
            hourly_values: Historical values grouped by hour
            hour: Current hour (0-23)
            current: Current value to check
            
        Returns:
            Anomaly score (0-1)
        """
        if hour not in hourly_values or len(hourly_values[hour]) < 3:
            return 0.0
        
        hour_values = hourly_values[hour]
        
        # Use z-score for this specific hour
        return self.zscore_anomaly_score(np.array(hour_values), current)
    
    def ensemble_anomaly_score(self, metric: str, values: np.ndarray, 
                               current: float, hour: int,
                               hourly_values: Dict[int, List[float]]) -> Tuple[float, Dict]:
        """
        Combine multiple anomaly detection methods into ensemble score.
        
        Args:
            metric: Metric name
            values: Historical values
            current: Current value
            hour: Current hour
            hourly_values: Historical values grouped by hour
            
        Returns:
            Tuple of (ensemble_score, method_scores)
        """
        method_scores = {
            'zscore': self.zscore_anomaly_score(values, current),
            'iqr': self.iqr_anomaly_score(values, current),
            'ma': self.moving_average_score(values, current),
            'ewma': self.ewma_anomaly_score(values, current),
            'pattern': self.time_pattern_score(hourly_values, hour, current)
        }
        
        # Calculate weighted ensemble score
        ensemble_score = 0.0
        for method, score in method_scores.items():
            weight = self.ensemble_weights.get(method, 0.2)
            ensemble_score += weight * score
        
        return ensemble_score, method_scores
    
    def analyze_trend(self, values: np.ndarray) -> Dict:
        """
        Analyze trend in historical data.
        
        Args:
            values: Historical values
            
        Returns:
            Dictionary with trend information
        """
        if len(values) < 10:
            return {'trend': 'stable', 'slope': 0, 'r_squared': 0}
        
        # Linear regression for trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        # Determine trend direction
        if abs(slope) < 0.01 * np.mean(values):  # Less than 1% change per period
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value
        }
    
    def predict_anomaly_likelihood(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Predict anomaly likelihood using ensemble method.
        
        Args:
            historical_data: Dictionary of metric DataFrames
            
        Returns:
            DataFrame with anomaly predictions for each hour
        """
        logger.info("Predicting anomaly likelihood using advanced methods...")
        
        predictions = []
        today = datetime.utcnow().date()
        
        for hour in range(self.time_slots):
            hour_predictions = {
                'timestamp': datetime.combine(today, datetime.min.time()) + timedelta(hours=hour),
                'hour': hour
            }
            
            total_risk = 0
            metric_details = {}
            
            for metric in self.metrics:
                if metric not in historical_data or historical_data[metric].empty:
                    continue
                
                df = historical_data[metric]
                values = df['value'].values
                
                # Group by hour for pattern analysis
                df_copy = df.copy()
                df_copy['hour'] = df_copy.index.hour
                hourly_values = df_copy.groupby('hour')['value'].apply(list).to_dict()
                
                # Get expected value for this hour (mean of historical values at this hour)
                if hour in hourly_values and hourly_values[hour]:
                    expected_value = np.mean(hourly_values[hour])
                else:
                    expected_value = np.mean(values)
                
                # Calculate ensemble anomaly score
                ensemble_score, method_scores = self.ensemble_anomaly_score(
                    metric, values, expected_value, hour, hourly_values
                )
                
                # Analyze trend
                trend_info = self.analyze_trend(values)
                
                # Adjust risk based on trend
                trend_adjustment = 0
                if trend_info['trend'] == 'increasing' and trend_info['r_squared'] > 0.5:
                    trend_adjustment = 0.1
                
                final_score = min(ensemble_score + trend_adjustment, 1.0)
                
                # Determine risk level
                if final_score > self.critical_threshold:
                    risk_level = 'critical'
                elif final_score > self.warning_threshold:
                    risk_level = 'warning'
                else:
                    risk_level = 'normal'
                
                metric_details[metric] = {
                    'risk_score': round(final_score, 3),
                    'risk_level': risk_level,
                    'method_scores': {k: round(v, 3) for k, v in method_scores.items()},
                    'trend': trend_info['trend'],
                    'trend_confidence': round(trend_info['r_squared'], 3),
                    'expected_value': round(expected_value, 2)
                }
                
                total_risk += final_score
            
            # Calculate overall risk
            overall_risk = total_risk / len(self.metrics) if self.metrics else 0
            
            if overall_risk > self.critical_threshold:
                overall_level = 'critical'
            elif overall_risk > self.warning_threshold:
                overall_level = 'warning'
            else:
                overall_level = 'normal'
            
            hour_predictions['overall_risk_score'] = round(overall_risk, 3)
            hour_predictions['overall_risk_level'] = overall_level
            hour_predictions['metrics'] = metric_details
            
            predictions.append(hour_predictions)
        
        df_predictions = pd.DataFrame(predictions)
        df_predictions.sort_values('hour', inplace=True)
        
        logger.info(f"Generated {len(df_predictions)} hourly predictions using advanced methods")
        
        return df_predictions
    
    def generate_report(self, predictions_df: pd.DataFrame) -> str:
        """
        Generate human-readable report of predictions.
        
        Args:
            predictions_df: DataFrame with prediction results
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("ADVANCED ANOMALY PREDICTION REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Lookback period: {self.lookback_days} days")
        report_lines.append(f"Prediction date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report_lines.append(f"Methods: Z-Score, IQR, Moving Average, EWMA, Time Pattern")
        report_lines.append("")
        
        # Summary
        critical_hours = predictions_df[predictions_df['overall_risk_level'] == 'critical']
        warning_hours = predictions_df[predictions_df['overall_risk_level'] == 'warning']
        
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total hours predicted: {len(predictions_df)}")
        report_lines.append(f"  Critical risk hours: {len(critical_hours)}")
        report_lines.append(f"  Warning risk hours: {len(warning_hours)}")
        report_lines.append("")
        
        # High-risk time slots
        high_risk = predictions_df[predictions_df['overall_risk_score'] >= self.warning_threshold]
        if not high_risk.empty:
            report_lines.append(f"HIGH-RISK TIME SLOTS (risk ≥ {self.warning_threshold}):")
            for _, row in high_risk.iterrows():
                hour_str = f"{row['hour']:02d}:00"
                report_lines.append(f"  {hour_str}: {row['overall_risk_level'].upper()} "
                                  f"(score: {row['overall_risk_score']:.3f})")
                
                # Show top contributing metrics
                metrics = row['metrics']
                top_metrics = sorted(metrics.items(), key=lambda x: x[1]['risk_score'], reverse=True)[:2]
                for metric, details in top_metrics:
                    if details['risk_score'] > 0.3:
                        report_lines.append(f"    → {metric}: {details['risk_score']:.3f} "
                                          f"(trend: {details['trend']}, "
                                          f"confidence: {details['trend_confidence']:.2f})")
            report_lines.append("")
        
        # Method comparison
        report_lines.append("METHOD CONTRIBUTION ANALYSIS:")
        method_totals = {method: 0 for method in ['zscore', 'iqr', 'ma', 'ewma', 'pattern']}
        method_counts = {method: 0 for method in ['zscore', 'iqr', 'ma', 'ewma', 'pattern']}
        
        for _, row in predictions_df.iterrows():
            for metric, details in row['metrics'].items():
                for method, score in details.get('method_scores', {}).items():
                    method_totals[method] += score
                    method_counts[method] += 1
        
        for method in method_totals:
            avg_score = method_totals[method] / method_counts[method] if method_counts[method] > 0 else 0
            report_lines.append(f"  {method.upper():10s}: avg score = {avg_score:.3f}, "
                              f"weight = {self.ensemble_weights.get(method, 0):.2f}")
        
        report_lines.append("")
        report_lines.append("RECOMMENDATIONS:")
        if len(critical_hours) > 0:
            report_lines.append("  1. CRITICAL: Immediate attention required for high-risk periods")
            report_lines.append("  2. Consider proactive scaling or resource allocation")
            report_lines.append("  3. Monitor trends for predictive maintenance")
        elif len(warning_hours) > 0:
            report_lines.append("  1. WARNING: Prepare for potential issues")
            report_lines.append("  2. Review trends and adjust thresholds if needed")
        else:
            report_lines.append("  1. System appears stable")
            report_lines.append("  2. Continue regular monitoring")
            report_lines.append("  3. Review ensemble weights for optimization")
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)


def main():
    """Main function for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced anomaly predictor')
    parser.add_argument('--config', default='config/config.yaml', help='Path to config file')
    parser.add_argument('--days', type=int, default=7, help='Lookback days for historical data')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = AdvancedPredictor(args.config, args.days)
    
    # Generate sample predictions
    print("Advanced Predictor initialized successfully")
    print(f"Methods: Z-Score, IQR, Moving Average, EWMA, Time Pattern")
    print(f"Ensemble weights: {predictor.ensemble_weights}")


if __name__ == '__main__':
    main()
