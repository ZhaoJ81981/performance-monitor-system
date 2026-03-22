#!/usr/bin/env python3
"""
Disk failure prediction collector for Telegraf exec input.
Reads SMART attributes and predicts disk failure probability.
"""

import json
import sys
import os
import subprocess
import re
from datetime import datetime
from pathlib import Path

def get_smart_attributes(device="/dev/sda"):
    """Read SMART attributes from a disk device."""
    try:
        # Try smartctl (from smartmontools)
        result = subprocess.run(
            ["sudo", "smartctl", "-A", device],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            # Try without sudo
            result = subprocess.run(
                ["smartctl", "-A", device],
                capture_output=True,
                text=True,
                timeout=10
            )
        
        attributes = {}
        
        # Parse smartctl output
        lines = result.stdout.split('\n')
        in_attributes = False
        
        for line in lines:
            if "ID#" in line and "ATTRIBUTE_NAME" in line:
                in_attributes = True
                continue
            
            if in_attributes and line.strip() and not line.startswith("==="):
                parts = re.split(r'\s+', line.strip())
                if len(parts) >= 10:
                    attr_id = parts[0]
                    attr_name = parts[1]
                    value = parts[3]
                    worst = parts[4]
                    threshold = parts[5]
                    raw_value = parts[9]
                    
                    # Convert to appropriate types
                    try:
                        attributes[attr_name.lower()] = {
                            "id": int(attr_id),
                            "value": int(value),
                            "worst": int(worst),
                            "threshold": int(threshold),
                            "raw": int(raw_value) if raw_value.isdigit() else raw_value
                        }
                    except (ValueError, IndexError):
                        continue
        
        return attributes
        
    except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
        print(f"Error reading SMART attributes: {e}", file=sys.stderr)
        return {}

def get_disk_usage():
    """Get disk usage information."""
    try:
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = re.split(r'\s+', lines[1])
            if len(parts) >= 6:
                return {
                    "filesystem": parts[0],
                    "size": parts[1],
                    "used": parts[2],
                    "available": parts[3],
                    "use_percent": int(parts[4].replace('%', '')),
                    "mounted": parts[5]
                }
    except Exception as e:
        print(f"Error getting disk usage: {e}", file=sys.stderr)
    
    return {}

def calculate_failure_probability(smart_attributes):
    """Calculate disk failure probability from SMART attributes."""
    if not smart_attributes:
        return 0.05  # Default low probability if no SMART data
    
    # Weighted factors based on critical SMART attributes
    factors = []
    weights = []
    
    # Reallocated sector count (critical)
    if 'reallocated_sector_ct' in smart_attributes:
        raw = smart_attributes['reallocated_sector_ct']['raw']
        if raw > 0:
            factors.append(min(raw / 100, 1.0))  # Normalize
            weights.append(0.4)
    
    # Current pending sector count
    if 'pending_sector_ct' in smart_attributes:
        raw = smart_attributes['pending_sector_ct']['raw']
        if raw > 0:
            factors.append(min(raw / 50, 1.0))
            weights.append(0.3)
    
    # Uncorrectable sector count
    if 'uncorrectable_errors' in smart_attributes:
        raw = smart_attributes['uncorrectable_errors']['raw']
        if raw > 0:
            factors.append(min(raw / 10, 1.0))
            weights.append(0.2)
    
    # Temperature (high temperature increases risk)
    if 'temperature_celsius' in smart_attributes:
        temp = smart_attributes['temperature_celsius']['raw']
        if isinstance(temp, (int, float)):
            if temp > 60:
                factors.append(min((temp - 60) / 40, 1.0))
                weights.append(0.1)
    
    # Power on hours (wear factor)
    if 'power_on_hours' in smart_attributes:
        poh = smart_attributes['power_on_hours']['raw']
        if isinstance(poh, (int, float)):
            if poh > 20000:  # 20k hours
                factors.append(min((poh - 20000) / 40000, 1.0))
                weights.append(0.1)
    
    # Calculate weighted probability
    if factors and weights:
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
            probability = sum(f * w for f, w in zip(factors, normalized_weights))
            return min(probability, 0.95)  # Cap at 95%
    
    return 0.05  # Default low probability

def generate_influx_line(device, smart_attributes, failure_probability):
    """Generate InfluxDB line protocol output."""
    timestamp_ns = int(datetime.utcnow().timestamp() * 1e9)
    
    tags = f"device={device},host={os.uname().nodename}"
    
    fields = []
    
    # Add failure probability
    fields.append(f"failure_probability={failure_probability}")
    
    # Add critical SMART attributes
    critical_attrs = [
        'reallocated_sector_ct',
        'pending_sector_ct', 
        'uncorrectable_errors',
        'temperature_celsius',
        'power_on_hours'
    ]
    
    for attr in critical_attrs:
        if attr in smart_attributes:
            raw = smart_attributes[attr]['raw']
            if isinstance(raw, (int, float)):
                fields.append(f"{attr}={raw}")
    
    # Add health status
    if failure_probability > 0.8:
        health_status = "critical"
    elif failure_probability > 0.5:
        health_status = "warning"
    else:
        health_status = "healthy"
    
    fields.append(f"health_status=\"{health_status}\"")
    
    # Combine fields
    fields_str = ",".join(fields)
    
    # Line protocol: measurement,tags fields timestamp
    return f"disk_failure,{tags} {fields_str} {timestamp_ns}"

def main():
    """Main function for Telegraf exec input."""
    # Check for device parameter
    device = "/dev/sda"
    if len(sys.argv) > 1:
        device = sys.argv[1]
    
    # Get SMART attributes
    smart_attrs = get_smart_attributes(device)
    
    # Get disk usage
    disk_usage = get_disk_usage()
    
    # Calculate failure probability
    failure_prob = calculate_failure_probability(smart_attrs)
    
    # Generate InfluxDB line protocol output
    influx_line = generate_influx_line(device, smart_attrs, failure_prob)
    
    # Output for Telegraf
    print(influx_line)
    
    # Also output JSON for debugging if requested
    if "--debug" in sys.argv:
        debug_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "device": device,
            "failure_probability": failure_prob,
            "smart_attributes": {k: v for k, v in list(smart_attrs.items())[:5]},  # First 5
            "disk_usage": disk_usage
        }
        print(json.dumps(debug_info, indent=2), file=sys.stderr)

if __name__ == "__main__":
    main()