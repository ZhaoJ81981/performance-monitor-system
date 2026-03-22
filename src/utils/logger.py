#!/usr/bin/env python3
"""
Centralized logging configuration for Performance Monitor System.
"""

import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yaml

# Try to load config
try:
    with open('config/local.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    config = {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': './logs/pms.log'
        }
    }

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data)

class InfluxDBHandler(logging.Handler):
    """Log handler that sends logs to InfluxDB."""
    
    def __init__(self, influx_client, measurement='logs'):
        super().__init__()
        self.influx_client = influx_client
        self.measurement = measurement
        self.write_api = influx_client.write_api()
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record to InfluxDB."""
        try:
            from influxdb_client import Point
            
            point = Point(self.measurement) \
                .tag("logger", record.name) \
                .tag("level", record.levelname) \
                .tag("module", record.module) \
                .tag("function", record.funcName) \
                .field("message", record.getMessage()) \
                .field("line", record.lineno)
            
            # Add exception if present
            if record.exc_info:
                point.field("exception", self.formatException(record.exc_info))
            
            # Write to InfluxDB
            self.write_api.write(
                bucket=config.get('data', {}).get('database', 'metrics'),
                org=config.get('data', {}).get('influxdb_org', 'pms'),
                record=point
            )
        except Exception:
            # Don't let logging errors crash the application
            pass

def setup_logging(
    name: str = "pms",
    level: str = None,
    log_file: str = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_influxdb: bool = False,
    influx_client = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_influxdb: Enable InfluxDB logging
        influx_client: InfluxDB client instance
        
    Returns:
        Configured logger instance
    """
    # Get configuration
    log_config = config.get('logging', {})
    
    if level is None:
        level = log_config.get('level', 'INFO')
    
    if log_file is None:
        log_file = log_config.get('file', './logs/pms.log')
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        fmt=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    json_formatter = JSONFormatter()
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if enable_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(json_formatter)  # JSON for file logs
        logger.addHandler(file_handler)
    
    # InfluxDB handler
    if enable_influxdb and influx_client and log_config.get('log_to_influxdb', False):
        influx_handler = InfluxDBHandler(influx_client, log_config.get('influxdb_measurement', 'logs'))
        influx_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.addHandler(influx_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str = "pms") -> logging.Logger:
    """
    Get or create a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

# Context manager for timing operations
class TimerContext:
    """Context manager for timing operations with logging."""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None):
        self.operation = operation
        self.logger = logger or get_logger("timer")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.info(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(
                f"Operation failed: {self.operation}",
                extra={
                    'operation': self.operation,
                    'elapsed_seconds': elapsed,
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val)
                }
            )
        else:
            self.logger.info(
                f"Completed operation: {self.operation}",
                extra={
                    'operation': self.operation,
                    'elapsed_seconds': elapsed
                }
            )
        return False  # Don't suppress exceptions

# Decorator for timing functions
def timed(operation: str = None):
    """
    Decorator to time function execution.
    
    Args:
        operation: Operation name (defaults to function name)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            with TimerContext(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Logging utilities for different components
def get_ml_logger() -> logging.Logger:
    """Get logger for ML pipeline."""
    return get_logger("pms.ml")

def get_api_logger() -> logging.Logger:
    """Get logger for API."""
    return get_logger("pms.api")

def get_collector_logger() -> logging.Logger:
    """Get logger for data collectors."""
    return get_logger("pms.collector")

def get_monitor_logger() -> logging.Logger:
    """Get logger for monitoring."""
    return get_logger("pms.monitor")

# Structured log helpers
def log_metric(
    logger: logging.Logger,
    metric_name: str,
    value: float,
    tags: Dict[str, Any] = None,
    level: str = "INFO"
):
    """Log a metric with structured format."""
    extra = {'metric': metric_name, 'value': value}
    if tags:
        extra.update({f'tag_{k}': v for k, v in tags.items()})
    
    log_method = getattr(logger, level.lower(), logger.info)
    log_method(f"Metric: {metric_name} = {value}", extra=extra)

def log_alert(
    logger: logging.Logger,
    alert_type: str,
    message: str,
    severity: str = "warning",
    details: Dict[str, Any] = None
):
    """Log an alert with structured format."""
    extra = {
        'alert_type': alert_type,
        'severity': severity,
        'timestamp': datetime.utcnow().isoformat()
    }
    if details:
        extra.update({f'detail_{k}': v for k, v in details.items()})
    
    log_method = getattr(logger, severity.lower(), logger.warning)
    log_method(f"Alert: {alert_type} - {message}", extra=extra)

def log_prediction(
    logger: logging.Logger,
    model: str,
    prediction: float,
    confidence: float,
    features: Dict[str, float] = None
):
    """Log a prediction with structured format."""
    extra = {
        'model': model,
        'prediction': prediction,
        'confidence': confidence,
        'timestamp': datetime.utcnow().isoformat()
    }
    if features:
        extra.update({f'feature_{k}': v for k, v in features.items()})
    
    logger.info(f"Prediction: {model} = {prediction:.4f} (confidence: {confidence:.2f})", extra=extra)

# Default logger setup
default_logger = setup_logging()

if __name__ == "__main__":
    # Test logging configuration
    logger = get_logger("test")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test structured logging
    log_metric(logger, "cpu_usage", 45.2, {"host": "localhost"})
    log_alert(logger, "high_cpu", "CPU usage exceeds threshold", "warning", {"usage": 90.5})
    log_prediction(logger, "lstm", 0.85, 0.92, {"feature1": 0.5, "feature2": 0.3})
    
    # Test timer context
    with TimerContext("test_operation", logger):
        import time
        time.sleep(0.1)
    
    print("Logging test completed. Check logs for output.")