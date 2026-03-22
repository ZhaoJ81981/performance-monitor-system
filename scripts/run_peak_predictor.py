#!/usr/bin/env python3
"""
Run peak-based anomaly predictor and generate reports.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ml_pipeline.peak_based_predictor import PeakBasedPredictor
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run peak-based anomaly predictor')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to look back for historical data')
    parser.add_argument('--output-dir', default='reports',
                       help='Directory to save report files')
    parser.add_argument('--save-to-influx', action='store_true', default=True,
                       help='Save predictions to InfluxDB')
    
    args = parser.parse_args()
    
    logger.info(f"Starting peak-based predictor with {args.days} days lookback")
    
    try:
        # Initialize predictor
        predictor = PeakBasedPredictor(args.config, args.days)
        
        # Generate predictions
        predictions_df = predictor.predict_anomaly_likelihood()
        
        # Generate report
        report = predictor.generate_report(predictions_df)
        
        # Print report to console
        print(report)
        
        # Save report to file
        os.makedirs(args.output_dir, exist_ok=True)
        report_filename = f"peak_prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(args.output_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to: {report_path}")
        
        # Save predictions to InfluxDB if requested
        if args.save_to_influx:
            predictor.save_predictions_to_influx(predictions_df)
            logger.info("Predictions saved to InfluxDB")
        
        # Generate JSON output for API consumption
        json_output = {
            'timestamp': datetime.now().isoformat(),
            'lookback_days': args.days,
            'total_hours_predicted': len(predictions_df),
            'critical_hours': len(predictions_df[predictions_df['overall_risk_level'] == 'critical']),
            'warning_hours': len(predictions_df[predictions_df['overall_risk_level'] == 'warning']),
            'high_risk_periods': []
        }
        
        # Add high-risk periods
        high_risk = predictions_df[predictions_df['overall_risk_score'] >= 0.5]
        for _, row in high_risk.iterrows():
            json_output['high_risk_periods'].append({
                'hour': int(row['hour']),
                'risk_score': float(row['overall_risk_score']),
                'risk_level': row['overall_risk_level']
            })
        
        # Save JSON output
        import json
        json_filename = f"peak_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = os.path.join(args.output_dir, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        logger.info(f"JSON output saved to: {json_path}")
        
        logger.info("Peak-based prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error running peak-based predictor: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()