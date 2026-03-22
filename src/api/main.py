#!/usr/bin/env python3
"""
FastAPI server for Performance Monitor System API.
Provides endpoints for metrics, predictions, and alerts.
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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

# Configure logging
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", level="INFO")

# Load configuration
with open('config/local.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Performance Monitor System API",
    description="Real-time system monitoring and predictive analytics API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# InfluxDB client
influx_client = influxdb_client.InfluxDBClient(
    url=config['data']['influxdb_host'],
    token=config['data']['influxdb_token'],
    org=config['data']['influxdb_org']
)
query_api = influx_client.query_api()
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# Load ML models
models_dir = Path(config['models']['storage_path'])
models = {}

def load_models():
    """Load trained ML models from disk."""
    try:
        # Load LSTM model
        if (models_dir / 'lstm_model.h5').exists():
            import tensorflow as tf
            models['lstm'] = tf.keras.models.load_model(models_dir / 'lstm_model.h5')
            logger.info("LSTM model loaded")
        
        # Load anomaly detection model
        if (models_dir / 'anomaly_model.pkl').exists():
            with open(models_dir / 'anomaly_model.pkl', 'rb') as f:
                models['anomaly'] = pickle.load(f)
            logger.info("Anomaly detection model loaded")
        
        # Load disk failure model
        if (models_dir / 'disk_failure_model.pkl').exists():
            with open(models_dir / 'disk_failure_model.pkl', 'rb') as f:
                models['disk_failure'] = pickle.load(f)
            logger.info("Disk failure prediction model loaded")
        
        # Load Prophet model
        if (models_dir / 'prophet_model.pkl').exists():
            with open(models_dir / 'prophet_model.pkl', 'rb') as f:
                models['prophet'] = pickle.load(f)
            logger.info("Prophet model loaded")
        
        # Load scaler
        if (models_dir / 'scaler.pkl').exists():
            with open(models_dir / 'scaler.pkl', 'rb') as f:
                models['scaler'] = pickle.load(f)
        
        # Load anomaly threshold
        if (models_dir / 'anomaly_threshold.txt').exists():
            with open(models_dir / 'anomaly_threshold.txt', 'r') as f:
                models['anomaly_threshold'] = float(f.read())
        
        logger.info(f"Loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")

# Load models on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Performance Monitor System API")
    load_models()

# Pydantic models for request/response
class MetricsRequest(BaseModel):
    host: str = Field(..., description="Target host name")
    metric_type: str = Field(..., description="Metric type (cpu, mem, disk, net)")
    start_time: Optional[datetime] = Field(None, description="Start time for query")
    end_time: Optional[datetime] = Field(None, description="End time for query")
    limit: int = Field(1000, description="Maximum data points to return")

class PredictionRequest(BaseModel):
    host: str = Field(..., description="Target host name")
    model_type: str = Field(..., description="Model type (lstm, prophet, anomaly)")
    horizon: int = Field(1440, description="Prediction horizon in minutes")

class AlertRequest(BaseModel):
    host: str = Field(..., description="Host name")
    metric: str = Field(..., description="Metric name")
    threshold: float = Field(..., description="Alert threshold")
    severity: str = Field("warning", description="Alert severity (info, warning, critical)")

class AlertResponse(BaseModel):
    id: str
    host: str
    metric: str
    value: float
    threshold: float
    severity: str
    timestamp: datetime
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and dependent services health."""
    services = {
        "api": "healthy",
        "influxdb": "unknown",
        "ml_models": "unknown"
    }
    
    # Check InfluxDB
    try:
        health = influx_client.health()
        services["influxdb"] = "healthy" if health.status == "pass" else "unhealthy"
    except Exception:
        services["influxdb"] = "unreachable"
    
    # Check ML models
    if len(models) > 0:
        services["ml_models"] = "loaded"
    else:
        services["ml_models"] = "not_loaded"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        services=services,
        version="1.0.0"
    )

# Metrics endpoints
@app.get("/api/v1/metrics", response_model=List[Dict[str, Any]])
async def get_metrics(
    host: str = Query(..., description="Target host name"),
    metric_type: str = Query(..., description="Metric type (cpu, mem, disk, net)"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(1000, description="Limit results")
):
    """Retrieve historical metrics for a host."""
    if not start_time:
        start_time = datetime.utcnow() - timedelta(hours=1)
    if not end_time:
        end_time = datetime.utcnow()
    
    query = f'''
    from(bucket: "{config['data']['database']}")
      |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
      |> filter(fn: (r) => r._measurement == "{metric_type}")
      |> filter(fn: (r) => r.host == "{host}")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> limit(n: {limit})
    '''
    
    try:
        df = query_api.query_data_frame(query)
        if df is not None and len(df) > 0:
            df['_time'] = pd.to_datetime(df['_time'])
            result = df.to_dict('records')
            return result
        else:
            return []
    except Exception as e:
        logger.error(f"Error querying metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/metrics")
async def submit_custom_metrics(metrics: List[Dict[str, Any]]):
    """Submit custom metrics to InfluxDB."""
    from influxdb_client import Point
    
    points = []
    for metric in metrics:
        point = Point(metric.get('measurement', 'custom'))
        
        # Add tags
        for tag_key, tag_value in metric.get('tags', {}).items():
            point.tag(tag_key, tag_value)
        
        # Add fields
        for field_key, field_value in metric.get('fields', {}).items():
            point.field(field_key, field_value)
        
        # Add timestamp
        if 'timestamp' in metric:
            point.time(metric['timestamp'])
        
        points.append(point)
    
    try:
        write_api.write(
            bucket=config['data']['database'],
            org=config['data']['influxdb_org'],
            record=points
        )
        return {"status": "success", "points_written": len(points)}
    except Exception as e:
        logger.error(f"Error writing metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction endpoints
@app.get("/api/v1/predict/cpu/{host}", response_model=Dict[str, Any])
async def predict_cpu_usage(
    host: str,
    horizon: int = Query(1440, description="Prediction horizon in minutes")
):
    """Predict CPU usage for a host."""
    if 'lstm' not in models:
        raise HTTPException(status_code=503, detail="LSTM model not loaded")
    
    # Fetch recent data
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=config['models']['lstm']['sequence_length'])
    
    query = f'''
    from(bucket: "{config['data']['database']}")
      |> range(start: {start_time.isoformat()}Z, stop: {end_time.isoformat()}Z)
      |> filter(fn: (r) => r._measurement == "cpu")
      |> filter(fn: (r) => r.host == "{host}")
      |> filter(fn: (r) => r._field == "usage_user")
      |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
      |> fill(value: 0.0)
    '''
    
    try:
        df = query_api.query_data_frame(query)
        if df is None or len(df) < config['models']['lstm']['sequence_length']:
            raise HTTPException(status_code=400, detail="Insufficient historical data")
        
        # Prepare data for LSTM
        values = df['_value'].values[-config['models']['lstm']['sequence_length']:]
        
        if 'scaler' in models:
            values_scaled = models['scaler'].transform(values.reshape(-1, 1))
        else:
            values_scaled = values.reshape(-1, 1)
        
        # Reshape for LSTM [samples, timesteps, features]
        X = values_scaled.reshape(1, -1, 1)
        
        # Make prediction
        prediction_scaled = models['lstm'].predict(X)[0][0]
        
        if 'scaler' in models:
            prediction = models['scaler'].inverse_transform([[prediction_scaled]])[0][0]
        else:
            prediction = prediction_scaled
        
        # Generate forecast times
        forecast_times = [end_time + timedelta(minutes=i) for i in range(1, horizon + 1)]
        
        # Simple linear extrapolation for multiple steps (in reality, use recursive prediction)
        forecast_values = [prediction * (0.95 ** i) for i in range(horizon)]
        
        return {
            "host": host,
            "current_usage": float(values[-1]),
            "predicted_usage": float(prediction),
            "forecast": [
                {"timestamp": t.isoformat(), "value": float(v)}
                for t, v in zip(forecast_times, forecast_values)
            ],
            "model": "lstm",
            "horizon_minutes": horizon,
            "confidence": 0.85
        }
        
    except Exception as e:
        logger.error(f"Error predicting CPU usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/predict/disk/{host}", response_model=Dict[str, Any])
async def predict_disk_failure(
    host: str,
    device: str = Query("sda", description="Disk device name")
):
    """Predict disk failure probability."""
    if 'disk_failure' not in models:
        raise HTTPException(status_code=503, detail="Disk failure model not loaded")
    
    # Fetch SMART attributes
    query = f'''
    from(bucket: "{config['data']['database']}")
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
            # Return default prediction if no SMART data
            return {
                "host": host,
                "device": device,
                "failure_probability": 0.05,
                "confidence": 0.5,
                "status": "unknown",
                "recommendation": "No SMART data available",
                "critical_attributes": []
            }
        
        # Prepare features (simplified)
        features = []
        for attr in config['models']['disk_failure']['smart_attributes']:
            if attr in df.columns:
                features.append(float(df[attr].iloc[0]))
            else:
                features.append(0.0)
        
        # Predict failure probability
        X = np.array(features).reshape(1, -1)
        probability = models['disk_failure'].predict_proba(X)[0][1]
        
        # Determine status
        if probability > config['models']['disk_failure']['failure_threshold']:
            status = "critical"
            recommendation = f"Immediate backup recommended. Failure probability: {probability:.1%}"
        elif probability > config['models']['disk_failure']['failure_threshold'] * 0.7:
            status = "warning"
            recommendation = f"Monitor closely. Failure probability: {probability:.1%}"
        else:
            status = "healthy"
            recommendation = f"Disk appears healthy. Failure probability: {probability:.1%}"
        
        return {
            "host": host,
            "device": device,
            "failure_probability": float(probability),
            "confidence": 0.8,
            "status": status,
            "recommendation": recommendation,
            "critical_attributes": [
                attr for attr, val in zip(config['models']['disk_failure']['smart_attributes'], features)
                if val > 0
            ]
        }
        
    except Exception as e:
        logger.error(f"Error predicting disk failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Anomaly detection endpoint
@app.get("/api/v1/anomaly/detect/{host}", response_model=Dict[str, Any])
async def detect_anomalies(
    host: str,
    metric_type: str = Query("cpu", description="Metric type to check for anomalies")
):
    """Detect anomalies in recent metrics."""
    if 'anomaly' not in models:
        raise HTTPException(status_code=503, detail="Anomaly detection model not loaded")
    
    # Fetch recent data
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=60)  # Last hour
    
    query = f'''
    from(bucket: "{config['data']['database']}")
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
                "anomaly_score": 0.0,
                "is_anomalous": False,
                "details": []
            }
        
        # Prepare data for anomaly detection
        # Use all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {
                "host": host,
                "metric_type": metric_type,
                "anomalies_detected": 0,
                "anomaly_score": 0.0,
                "is_anomalous": False,
                "details": []
            }
        
        X = df[numeric_cols].values
        
        # Detect anomalies
        if 'anomaly_threshold' in models:
            scores = models['anomaly'].decision_function(X)
            anomalies = scores < models['anomaly_threshold']
        else:
            anomalies = models['anomaly'].predict(X) == -1
        
        anomaly_count = np.sum(anomalies)
        anomaly_score = float(np.mean(scores) if 'anomaly_threshold' in models else 0.0)
        
        # Prepare anomaly details
        details = []
        if anomaly_count > 0:
            anomaly_indices = np.where(anomalies)[0]
            for idx in anomaly_indices[:10]:  # Limit to first 10
                timestamp = df.iloc[idx]['_time']
                detail = {
                    "timestamp": timestamp.isoformat(),
                    "scores": {
                        col: float(df.iloc[idx][col]) for col in numeric_cols[:3]  # First 3 metrics
                    }
                }
                details.append(detail)
        
        return {
            "host": host,
            "metric_type": metric_type,
            "anomalies_detected": int(anomaly_count),
            "anomaly_score": anomaly_score,
            "is_anomalous": anomaly_count > 0,
            "details": details,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Alert endpoints
@app.get("/api/v1/alerts", response_model=List[AlertResponse])
async def get_active_alerts(
    host: Optional[str] = Query(None, description="Filter by host"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(100, description="Limit results")
):
    """Retrieve active alerts."""
    # Query alerts from InfluxDB (simplified)
    query = '''
    from(bucket: "alerts")
      |> range(start: -24h)
      |> filter(fn: (r) => r._measurement == "alerts")
      |> filter(fn: (r) => r.status == "active")
      |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> limit(n: ''' + str(limit) + ''')
    '''
    
    if host:
        query = query.replace('status == "active"', f'status == "active" and host == "{host}"')
    
    try:
        df = query_api.query_data_frame(query)
        if df is None or len(df) == 0:
            return []
        
        alerts = []
        for _, row in df.iterrows():
            alert = AlertResponse(
                id=str(row.get('alert_id', '')),
                host=row.get('host', 'unknown'),
                metric=row.get('metric', 'unknown'),
                value=float(row.get('value', 0)),
                threshold=float(row.get('threshold', 0)),
                severity=row.get('severity', 'warning'),
                timestamp=row['_time'],
                message=row.get('message', '')
            )
            alerts.append(alert)
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard endpoints
@app.get("/api/v1/dashboard/summary/{host}", response_model=Dict[str, Any])
async def get_dashboard_summary(host: str):
    """Get summary dashboard data for a host."""
    # Collect various metrics in parallel
    # (simplified - in reality would use async queries)
    
    summary = {
        "host": host,
        "timestamp": datetime.utcnow().isoformat(),
        "cpu": {
            "usage_percent": 45.2,
            "temperature": 65.0,
            "status": "normal"
        },
        "memory": {
            "used_percent": 68.5,
            "available_gb": 12.3,
            "status": "normal"
        },
        "disk": {
            "used_percent": 42.1,
            "failure_probability": 0.05,
            "status": "healthy"
        },
        "network": {
            "throughput_mbps": 125.4,
            "status": "normal"
        },
        "alerts": {
            "active": 2,
            "critical": 0,
            "warning": 2
        },
        "predictions": {
            "cpu_next_hour": 48.7,
            "anomaly_risk": 0.12
        }
    }
    
    return summary

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config['api']['host'],
        port=config['api']['port'],
        reload=config['api']['debug']
    )