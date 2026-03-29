#!/usr/bin/env python3
"""
FastAPI server v2 for Performance Monitor System API.
Improvements:
- Rate limiting middleware
- In-memory caching layer
- Enhanced health checks
- Better error handling
- Async database operations
- Request validation
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import yaml
import json
import asyncio
from datetime import datetime, timedelta
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging
from loguru import logger
import sys
import traceback

# Import custom utilities
from src.utils.cache import metrics_cache, prediction_cache, alert_cache, cached
from src.utils.rate_limiter import RateLimitMiddleware, RateLimiter

# Configure logging
logger.remove()
logger.add(
    sys.stdout, 
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}", 
    level="INFO"
)

# Load configuration
config_path = Path('config/local.yaml')
if not config_path.exists():
    config_path = Path('config/config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app with enhanced metadata
app = FastAPI(
    title="Performance Monitor System API",
    description="""
Real-time system monitoring and predictive analytics API.

## Features
- **Metrics**: Collect and query system metrics
- **Predictions**: ML-based forecasting
- **Anomaly Detection**: Automatic anomaly identification
- **Alerts**: Configurable alerting

## Rate Limits
- 120 requests/minute per client
- Burst allowance: 20 requests
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "health", "description": "Health check endpoints"},
        {"name": "metrics", "description": "Metrics collection and retrieval"},
        {"name": "predictions", "description": "ML-based predictions"},
        {"name": "anomaly", "description": "Anomaly detection"},
        {"name": "alerts", "description": "Alert management"},
        {"name": "dashboard", "description": "Dashboard data"},
    ]
)

# Rate limiter
rate_limiter = RateLimiter(requests_per_minute=120, burst_size=20)

# Add middlewares
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if config.get('api', {}).get('debug', False) else "An unexpected error occurred",
            "path": str(request.url.path),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# InfluxDB client with connection pooling
influx_config = config.get('data', {})
influx_client = influxdb_client.InfluxDBClient(
    url=influx_config.get('influxdb_host', 'http://localhost:8086'),
    token=influx_config.get('influxdb_token', ''),
    org=influx_config.get('influxdb_org', 'default'),
    timeout=30_000,  # 30s timeout
    enable_gzip=True
)
query_api = influx_client.query_api()
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Load ML models
models_dir = Path(config.get('models', {}).get('storage_path', 'models'))
models = {}
models_loaded_at = None

def load_models():
    """Load trained ML models from disk with error handling."""
    global models_loaded_at
    loaded = []
    failed = []
    
    try:
        # Load LSTM model
        lstm_path = models_dir / 'lstm_model.h5'
        if lstm_path.exists():
            try:
                import tensorflow as tf
                models['lstm'] = tf.keras.models.load_model(lstm_path)
                loaded.append('lstm')
            except Exception as e:
                failed.append(('lstm', str(e)))
        
        # Load anomaly detection model
        anomaly_path = models_dir / 'anomaly_model.pkl'
        if anomaly_path.exists():
            try:
                with open(anomaly_path, 'rb') as f:
                    models['anomaly'] = pickle.load(f)
                loaded.append('anomaly')
            except Exception as e:
                failed.append(('anomaly', str(e)))
        
        # Load disk failure model
        disk_path = models_dir / 'disk_failure_model.pkl'
        if disk_path.exists():
            try:
                with open(disk_path, 'rb') as f:
                    models['disk_failure'] = pickle.load(f)
                loaded.append('disk_failure')
            except Exception as e:
                failed.append(('disk_failure', str(e)))
        
        # Load Prophet model
        prophet_path = models_dir / 'prophet_model.pkl'
        if prophet_path.exists():
            try:
                with open(prophet_path, 'rb') as f:
                    models['prophet'] = pickle.load(f)
                loaded.append('prophet')
            except Exception as e:
                failed.append(('prophet', str(e)))
        
        # Load scaler and threshold
        if (models_dir / 'scaler.pkl').exists():
            with open(models_dir / 'scaler.pkl', 'rb') as f:
                models['scaler'] = pickle.load(f)
        
        if (models_dir / 'anomaly_threshold.txt').exists():
            with open(models_dir / 'anomaly_threshold.txt', 'r') as f:
                models['anomaly_threshold'] = float(f.read())
        
        models_loaded_at = datetime.utcnow()
        logger.info(f"Models loaded: {loaded}")
        if failed:
            logger.warning(f"Models failed to load: {failed}")
        
        return loaded, failed
        
    except Exception as e:
        logger.error(f"Error in model loading: {e}")
        return loaded, failed

# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Performance Monitor System API v2")
    loaded, failed = load_models()
    logger.info(f"Startup complete. Models: {len(loaded)} loaded, {len(failed)} failed")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API")
    influx_client.close()


# ============= Pydantic Models =============

class MetricsQuery(BaseModel):
    host: str = Field(..., description="Target host name", min_length=1, max_length=255)
    metric_type: str = Field(..., description="Metric type", pattern="^(cpu|mem|disk|net|system)$")
    start_time: Optional[datetime] = Field(None, description="Start time for query")
    end_time: Optional[datetime] = Field(None, description="End time for query")
    limit: int = Field(1000, ge=1, le=10000, description="Maximum data points")
    
    @validator('start_time', 'end_time', pre=True)
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

class PredictionRequest(BaseModel):
    host: str = Field(..., min_length=1, max_length=255)
    model_type: str = Field("lstm", pattern="^(lstm|prophet|ensemble)$")
    horizon: int = Field(1440, ge=1, le=10080, description="Prediction horizon in minutes (max 7 days)")

class AlertConfig(BaseModel):
    host: str = Field(..., min_length=1)
    metric: str = Field(...)
    threshold: float = Field(...)
    operator: str = Field("gt", pattern="^(gt|lt|eq|gte|lte)$")
    severity: str = Field("warning", pattern="^(info|warning|critical)$")
    cooldown_minutes: int = Field(5, ge=1, le=1440)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    services: Dict[str, Dict[str, Any]]
    cache_stats: Dict[str, Dict[str, Any]]
    rate_limiter_stats: Dict[str, Any]


# ============= Health Endpoints =============

startup_time = datetime.utcnow()

@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Comprehensive health check including all dependent services.
    
    Returns detailed status of:
    - API server
    - InfluxDB connection
    - ML models
    - Cache statistics
    - Rate limiter status
    """
    services = {}
    
    # Check InfluxDB
    try:
        health = influx_client.health()
        services["influxdb"] = {
            "status": "healthy" if health.status == "pass" else "unhealthy",
            "version": getattr(health, 'version', 'unknown'),
            "message": getattr(health, 'message', '')
        }
    except Exception as e:
        services["influxdb"] = {
            "status": "unreachable",
            "error": str(e)
        }
    
    # Check ML models
    services["ml_models"] = {
        "status": "loaded" if models else "not_loaded",
        "loaded_models": list(models.keys()),
        "loaded_at": models_loaded_at.isoformat() if models_loaded_at else None
    }
    
    # Determine overall status
    overall_status = "healthy"
    if services["influxdb"]["status"] != "healthy":
        overall_status = "degraded"
    if not models:
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        uptime_seconds=(datetime.utcnow() - startup_time).total_seconds(),
        version="2.0.0",
        services=services,
        cache_stats={
            "metrics": metrics_cache.stats,
            "predictions": prediction_cache.stats,
            "alerts": alert_cache.stats
        },
        rate_limiter_stats=rate_limiter.stats
    )

@app.get("/health/live", tags=["health"])
async def liveness_probe():
    """Kubernetes liveness probe - just confirms API is running."""
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/ready", tags=["health"])
async def readiness_probe():
    """Kubernetes readiness probe - confirms dependencies are available."""
    try:
        influx_client.health()
        return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
    except Exception:
        raise HTTPException(status_code=503, detail="Database not ready")


# ============= Metrics Endpoints =============

@app.get("/api/v2/metrics", tags=["metrics"])
@cached(metrics_cache, ttl=30)
async def get_metrics(
    host: str = Query(..., min_length=1, max_length=255),
    metric_type: str = Query(..., regex="^(cpu|mem|disk|net|system)$"),
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(1000, ge=1, le=10000),
    aggregation: str = Query("mean", regex="^(mean|max|min|sum|count)$"),
    window: str = Query("1m", regex="^[0-9]+[smh]$")
):
    """
    Retrieve historical metrics for a host with aggregation.
    
    - **host**: Target hostname
    - **metric_type**: One of cpu, mem, disk, net, system
    - **aggregation**: Aggregation function (mean, max, min, sum, count)
    - **window**: Aggregation window (e.g., 1m, 5m, 1h)
    """
    if not start_time:
        start_time = datetime.utcnow() - timedelta(hours=1)
    if not end_time:
        end_time = datetime.utcnow()
    
    database = influx_config.get('database', 'metrics')
    
    query = f'''
    from(bucket: "{database}")
      |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
      |> filter(fn: (r) => r._measurement == "{metric_type}")
      |> filter(fn: (r) => r.host == "{host}")
      |> aggregateWindow(every: {window}, fn: {aggregation}, createEmpty: false)
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> limit(n: {limit})
    '''
    
    try:
        df = query_api.query_data_frame(query)
        if df is not None and len(df) > 0:
            # Clean up DataFrame
            df['_time'] = pd.to_datetime(df['_time'])
            # Remove internal columns
            drop_cols = [c for c in df.columns if c.startswith('_') and c != '_time']
            df = df.drop(columns=drop_cols, errors='ignore')
            df = df.rename(columns={'_time': 'timestamp'})
            
            return {
                "host": host,
                "metric_type": metric_type,
                "count": len(df),
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "data": df.to_dict('records')
            }
        else:
            return {
                "host": host,
                "metric_type": metric_type,
                "count": 0,
                "time_range": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "data": []
            }
    except Exception as e:
        logger.error(f"Error querying metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/v2/metrics", tags=["metrics"])
async def submit_metrics(metrics: List[Dict[str, Any]], background_tasks: BackgroundTasks):
    """
    Submit custom metrics to the database.
    
    Accepts a list of metric objects with:
    - measurement: Metric name
    - tags: Dictionary of tags
    - fields: Dictionary of field values
    - timestamp: Optional ISO timestamp
    """
    if len(metrics) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 metrics per request")
    
    from influxdb_client import Point
    
    points = []
    errors = []
    
    for i, metric in enumerate(metrics):
        try:
            if 'measurement' not in metric or 'fields' not in metric:
                errors.append({"index": i, "error": "Missing required fields"})
                continue
            
            point = Point(metric['measurement'])
            
            for tag_key, tag_value in metric.get('tags', {}).items():
                point.tag(str(tag_key), str(tag_value))
            
            for field_key, field_value in metric['fields'].items():
                point.field(str(field_key), field_value)
            
            if 'timestamp' in metric:
                point.time(metric['timestamp'])
            
            points.append(point)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    
    if points:
        try:
            database = influx_config.get('database', 'metrics')
            org = influx_config.get('influxdb_org', 'default')
            write_api.write(bucket=database, org=org, record=points)
            
            # Invalidate related cache entries
            background_tasks.add_task(metrics_cache.clear)
            
        except Exception as e:
            logger.error(f"Error writing metrics: {e}")
            raise HTTPException(status_code=500, detail=f"Write failed: {str(e)}")
    
    return {
        "status": "success",
        "points_written": len(points),
        "errors": errors if errors else None
    }


# ============= Prediction Endpoints =============

@app.get("/api/v2/predict/cpu/{host}", tags=["predictions"])
@cached(prediction_cache, ttl=300)
async def predict_cpu_usage(
    host: str,
    horizon: int = Query(60, ge=1, le=1440, description="Prediction horizon in minutes")
):
    """
    Predict CPU usage for a host using LSTM model.
    
    Returns:
    - Current usage
    - Predicted usage
    - Forecast array with timestamps
    - Confidence score
    """
    if 'lstm' not in models:
        raise HTTPException(
            status_code=503, 
            detail="LSTM model not available. Please ensure models are trained and loaded."
        )
    
    sequence_length = config.get('models', {}).get('lstm', {}).get('sequence_length', 60)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=sequence_length * 2)  # Extra buffer
    
    database = influx_config.get('database', 'metrics')
    
    query = f'''
    from(bucket: "{database}")
      |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
      |> filter(fn: (r) => r._measurement == "cpu")
      |> filter(fn: (r) => r.host == "{host}")
      |> filter(fn: (r) => r._field == "usage_user" or r._field == "usage_system")
      |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    try:
        df = query_api.query_data_frame(query)
        
        if df is None or len(df) < sequence_length:
            raise HTTPException(
                status_code=400, 
                detail=f"Insufficient data. Need {sequence_length} points, got {len(df) if df is not None else 0}"
            )
        
        # Combine user and system CPU
        if 'usage_user' in df.columns and 'usage_system' in df.columns:
            values = (df['usage_user'] + df['usage_system']).values[-sequence_length:]
        elif 'usage_user' in df.columns:
            values = df['usage_user'].values[-sequence_length:]
        else:
            raise HTTPException(status_code=400, detail="CPU usage fields not found")
        
        # Scale if scaler available
        if 'scaler' in models:
            values_scaled = models['scaler'].transform(values.reshape(-1, 1))
        else:
            values_scaled = values.reshape(-1, 1)
        
        # Reshape for LSTM [samples, timesteps, features]
        X = values_scaled.reshape(1, -1, 1)
        
        # Predict
        prediction_scaled = models['lstm'].predict(X, verbose=0)[0][0]
        
        if 'scaler' in models:
            prediction = models['scaler'].inverse_transform([[prediction_scaled]])[0][0]
        else:
            prediction = prediction_scaled
        
        # Generate forecast (simplified multi-step)
        forecast_times = [end_time + timedelta(minutes=i) for i in range(1, horizon + 1)]
        
        # Decay factor for longer horizons (uncertainty increases)
        decay = 0.98
        forecast_values = []
        current_pred = prediction
        for i in range(horizon):
            forecast_values.append(max(0, min(100, current_pred)))
            current_pred = current_pred * decay + (values.mean() * (1 - decay))
        
        return {
            "host": host,
            "model": "lstm",
            "current_usage": round(float(values[-1]), 2),
            "predicted_usage": round(float(prediction), 2),
            "horizon_minutes": horizon,
            "confidence": 0.85 - (horizon / 1440 * 0.3),  # Confidence decreases with horizon
            "forecast": [
                {"timestamp": t.isoformat(), "value": round(v, 2)}
                for t, v in zip(forecast_times[:min(60, horizon)], forecast_values[:min(60, horizon)])
            ],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/v2/predict/disk/{host}", tags=["predictions"])
@cached(prediction_cache, ttl=600)
async def predict_disk_failure(
    host: str,
    device: str = Query("sda", description="Disk device name")
):
    """
    Predict disk failure probability based on SMART attributes.
    """
    if 'disk_failure' not in models:
        # Return mock prediction if model not available
        return {
            "host": host,
            "device": device,
            "failure_probability": 0.05,
            "confidence": 0.5,
            "status": "unknown",
            "recommendation": "Disk failure model not loaded. Using default estimation.",
            "smart_available": False,
            "generated_at": datetime.utcnow().isoformat()
        }
    
    # Query SMART attributes
    database = influx_config.get('database', 'metrics')
    
    query = f'''
    from(bucket: "{database}")
      |> range(start: -1h)
      |> filter(fn: (r) => r._measurement == "smart")
      |> filter(fn: (r) => r.host == "{host}")
      |> filter(fn: (r) => r.device == "{device}")
      |> last()
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    try:
        df = query_api.query_data_frame(query)
        
        if df is None or len(df) == 0:
            return {
                "host": host,
                "device": device,
                "failure_probability": 0.05,
                "confidence": 0.5,
                "status": "unknown",
                "recommendation": "No SMART data available for this device",
                "smart_available": False,
                "generated_at": datetime.utcnow().isoformat()
            }
        
        # Get SMART attributes from config
        smart_attrs = config.get('models', {}).get('disk_failure', {}).get(
            'smart_attributes', 
            ['reallocated_sectors', 'spin_retry_count', 'reallocation_events', 
             'current_pending_sectors', 'uncorrectable_sectors', 'command_timeout']
        )
        
        features = []
        for attr in smart_attrs:
            if attr in df.columns:
                features.append(float(df[attr].iloc[0]))
            else:
                features.append(0.0)
        
        X = np.array(features).reshape(1, -1)
        probability = float(models['disk_failure'].predict_proba(X)[0][1])
        
        failure_threshold = config.get('models', {}).get('disk_failure', {}).get('failure_threshold', 0.7)
        
        if probability > failure_threshold:
            status = "critical"
            recommendation = f"URGENT: Disk failure imminent ({probability:.1%}). Backup immediately."
        elif probability > failure_threshold * 0.7:
            status = "warning"
            recommendation = f"Elevated failure risk ({probability:.1%}). Schedule replacement."
        else:
            status = "healthy"
            recommendation = f"Disk appears healthy ({probability:.1%} failure probability)."
        
        return {
            "host": host,
            "device": device,
            "failure_probability": round(probability, 4),
            "confidence": 0.85,
            "status": status,
            "recommendation": recommendation,
            "smart_available": True,
            "smart_attributes_used": [a for a, f in zip(smart_attrs, features) if f > 0],
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Disk prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Anomaly Endpoints =============

@app.get("/api/v2/anomaly/detect/{host}", tags=["anomaly"])
@cached(metrics_cache, ttl=60)
async def detect_anomalies(
    host: str,
    metric_type: str = Query("cpu", regex="^(cpu|mem|disk|net)$"),
    lookback_minutes: int = Query(60, ge=5, le=1440)
):
    """
    Detect anomalies in recent metrics using Isolation Forest.
    """
    if 'anomaly' not in models:
        raise HTTPException(status_code=503, detail="Anomaly detection model not loaded")
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=lookback_minutes)
    database = influx_config.get('database', 'metrics')
    
    query = f'''
    from(bucket: "{database}")
      |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
      |> filter(fn: (r) => r._measurement == "{metric_type}")
      |> filter(fn: (r) => r.host == "{host}")
      |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    
    try:
        df = query_api.query_data_frame(query)
        
        if df is None or len(df) == 0:
            return {
                "host": host,
                "metric_type": metric_type,
                "anomalies_detected": 0,
                "is_anomalous": False,
                "lookback_minutes": lookback_minutes,
                "details": [],
                "generated_at": datetime.utcnow().isoformat()
            }
        
        # Prepare numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return {
                "host": host,
                "metric_type": metric_type,
                "anomalies_detected": 0,
                "is_anomalous": False,
                "error": "No numeric columns found",
                "generated_at": datetime.utcnow().isoformat()
            }
        
        X = df[numeric_cols].values
        
        # Detect anomalies
        threshold = models.get('anomaly_threshold', -0.5)
        scores = models['anomaly'].decision_function(X)
        anomalies = scores < threshold
        
        anomaly_count = int(np.sum(anomalies))
        
        # Get anomaly details
        details = []
        if anomaly_count > 0:
            anomaly_indices = np.where(anomalies)[0]
            for idx in anomaly_indices[:20]:  # Limit details
                row = df.iloc[idx]
                details.append({
                    "timestamp": row['_time'].isoformat() if '_time' in row else None,
                    "score": round(float(scores[idx]), 4),
                    "values": {col: round(float(row[col]), 2) for col in numeric_cols[:5]}
                })
        
        return {
            "host": host,
            "metric_type": metric_type,
            "anomalies_detected": anomaly_count,
            "total_points": len(df),
            "anomaly_rate": round(anomaly_count / len(df) * 100, 2),
            "is_anomalous": anomaly_count > 0,
            "mean_score": round(float(np.mean(scores)), 4),
            "lookback_minutes": lookback_minutes,
            "details": details,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= Dashboard Endpoints =============

@app.get("/api/v2/dashboard/summary/{host}", tags=["dashboard"])
@cached(metrics_cache, ttl=30)
async def get_dashboard_summary(host: str):
    """
    Get comprehensive dashboard summary for a host.
    Combines metrics, predictions, and alerts.
    """
    summary = {
        "host": host,
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {},
        "predictions": {},
        "alerts": {"active": 0, "critical": 0, "warning": 0},
        "health_score": 100
    }
    
    # Fetch latest metrics (simplified - in production would be parallel queries)
    database = influx_config.get('database', 'metrics')
    
    for metric in ['cpu', 'mem', 'disk']:
        query = f'''
        from(bucket: "{database}")
          |> range(start: -5m)
          |> filter(fn: (r) => r._measurement == "{metric}")
          |> filter(fn: (r) => r.host == "{host}")
          |> last()
        '''
        
        try:
            df = query_api.query_data_frame(query)
            if df is not None and len(df) > 0:
                value = df['_value'].mean()
                summary["metrics"][metric] = {
                    "value": round(float(value), 2),
                    "unit": "%" if metric in ['cpu', 'mem'] else "GB",
                    "status": "normal" if value < 80 else ("warning" if value < 90 else "critical")
                }
                
                # Adjust health score
                if value > 90:
                    summary["health_score"] -= 20
                elif value > 80:
                    summary["health_score"] -= 10
        except Exception:
            summary["metrics"][metric] = {"value": None, "status": "unknown"}
    
    summary["health_score"] = max(0, summary["health_score"])
    
    return summary


# ============= Admin Endpoints =============

@app.post("/api/v2/admin/cache/clear", tags=["admin"])
async def clear_cache(cache_type: str = Query("all", regex="^(all|metrics|predictions|alerts)$")):
    """Clear cache entries. Requires admin access in production."""
    if cache_type == "all":
        metrics_cache.clear()
        prediction_cache.clear()
        alert_cache.clear()
    elif cache_type == "metrics":
        metrics_cache.clear()
    elif cache_type == "predictions":
        prediction_cache.clear()
    elif cache_type == "alerts":
        alert_cache.clear()
    
    return {"status": "cleared", "cache_type": cache_type}

@app.post("/api/v2/admin/models/reload", tags=["admin"])
async def reload_models():
    """Reload ML models from disk."""
    global models
    models = {}
    loaded, failed = load_models()
    
    return {
        "status": "reloaded",
        "loaded": loaded,
        "failed": failed,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    api_config = config.get('api', {})
    uvicorn.run(
        "main_v2:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('debug', False),
        workers=api_config.get('workers', 1)
    )
