"""Monitoring and logging system for HPI pipeline"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import logging
import time
import threading
from collections import deque, defaultdict
import warnings


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric"""
    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class HealthStatus:
    """System health status"""
    status: str  # "healthy", "degraded", "unhealthy"
    components: Dict[str, str]  # Component -> status
    last_check: datetime
    issues: List[str] = field(default_factory=list)


class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector
        
        Args:
            retention_hours: Hours to retain metrics
        """
        self.retention_hours = retention_hours
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregates: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()
        
    def record_metric(self, metric: Metric):
        """Record a metric
        
        Args:
            metric: Metric to record
        """
        with self._lock:
            key = f"{metric.name}:{json.dumps(metric.labels, sort_keys=True)}"
            self.metrics[key].append(metric)
            
            # Update aggregates
            if metric.type == MetricType.COUNTER:
                self.aggregates[key]['total'] = self.aggregates[key].get('total', 0) + metric.value
            elif metric.type == MetricType.GAUGE:
                self.aggregates[key]['latest'] = metric.value
            elif metric.type == MetricType.HISTOGRAM:
                if 'values' not in self.aggregates[key]:
                    self.aggregates[key]['values'] = []
                self.aggregates[key]['values'].append(metric.value)
                
    def increment_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric
        
        Args:
            name: Metric name
            value: Increment value
            labels: Optional labels
        """
        self.record_metric(Metric(
            name=name,
            type=MetricType.COUNTER,
            value=value,
            labels=labels or {}
        ))
        
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Optional labels
        """
        self.record_metric(Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {}
        ))
        
    def record_time(self, name: str, duration_seconds: float, labels: Optional[Dict[str, str]] = None):
        """Record a timing metric
        
        Args:
            name: Metric name
            duration_seconds: Duration in seconds
            labels: Optional labels
        """
        self.record_metric(Metric(
            name=name,
            type=MetricType.TIMER,
            value=duration_seconds,
            labels=labels or {},
            unit="seconds"
        ))
        
    def get_metrics_summary(self, metric_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics summary
        
        Args:
            metric_name: Optional metric name filter
            
        Returns:
            Summary of metrics
        """
        with self._lock:
            summary = {}
            
            for key, metrics in self.metrics.items():
                if metric_name and not key.startswith(metric_name):
                    continue
                    
                # Clean old metrics
                cutoff = datetime.now() - timedelta(hours=self.retention_hours)
                recent_metrics = [m for m in metrics if m.timestamp > cutoff]
                
                if not recent_metrics:
                    continue
                    
                metric = recent_metrics[0]
                summary[key] = {
                    'type': metric.type.value,
                    'count': len(recent_metrics),
                    'latest': recent_metrics[-1].value,
                    'latest_timestamp': recent_metrics[-1].timestamp.isoformat()
                }
                
                # Add aggregates
                if metric.type == MetricType.COUNTER:
                    summary[key]['total'] = self.aggregates[key].get('total', 0)
                elif metric.type == MetricType.HISTOGRAM or metric.type == MetricType.TIMER:
                    values = [m.value for m in recent_metrics]
                    if values:
                        import numpy as np
                        summary[key].update({
                            'min': np.min(values),
                            'max': np.max(values),
                            'mean': np.mean(values),
                            'p50': np.percentile(values, 50),
                            'p95': np.percentile(values, 95),
                            'p99': np.percentile(values, 99)
                        })
                        
            return summary
            
    def clear_old_metrics(self):
        """Clear metrics older than retention period"""
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=self.retention_hours)
            
            for key in list(self.metrics.keys()):
                # Filter old metrics
                self.metrics[key] = deque(
                    (m for m in self.metrics[key] if m.timestamp > cutoff),
                    maxlen=10000
                )
                
                # Remove empty keys
                if not self.metrics[key]:
                    del self.metrics[key]
                    if key in self.aggregates:
                        del self.aggregates[key]


class AlertManager:
    """Manage system alerts"""
    
    def __init__(self, max_alerts: int = 1000):
        """Initialize alert manager
        
        Args:
            max_alerts: Maximum alerts to retain
        """
        self.max_alerts = max_alerts
        self.alerts: deque = deque(maxlen=max_alerts)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler
        
        Args:
            handler: Function to call on new alerts
        """
        self.alert_handlers.append(handler)
        
    def raise_alert(self, alert: Alert):
        """Raise an alert
        
        Args:
            alert: Alert to raise
        """
        with self._lock:
            self.alerts.append(alert)
            
            # Track active alerts
            if not alert.resolved:
                self.active_alerts[alert.alert_id] = alert
                
        # Call handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler error: {str(e)}")
                
    def resolve_alert(self, alert_id: str):
        """Resolve an alert
        
        Args:
            alert_id: Alert ID to resolve
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self.active_alerts[alert_id]
                
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get active alerts
        
        Args:
            level: Optional level filter
            
        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = list(self.active_alerts.values())
            
            if level:
                alerts = [a for a in alerts if a.level == level]
                
            return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
            
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        with self._lock:
            active_by_level = defaultdict(int)
            for alert in self.active_alerts.values():
                active_by_level[alert.level.value] += 1
                
            return {
                'total_alerts': len(self.alerts),
                'active_alerts': len(self.active_alerts),
                'active_by_level': dict(active_by_level),
                'oldest_active': min(
                    (a.timestamp for a in self.active_alerts.values()),
                    default=None
                )
            }


class PipelineMonitor:
    """Monitor pipeline execution and health"""
    
    def __init__(self,
                 metrics_collector: Optional[MetricsCollector] = None,
                 alert_manager: Optional[AlertManager] = None):
        """Initialize pipeline monitor
        
        Args:
            metrics_collector: Metrics collector instance
            alert_manager: Alert manager instance
        """
        self.metrics = metrics_collector or MetricsCollector()
        self.alerts = alert_manager or AlertManager()
        self.logger = logging.getLogger("pipeline_monitor")
        
        # Component health checks
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self._health_status = HealthStatus(
            status="healthy",
            components={},
            last_check=datetime.now()
        )
        
        # Thresholds for alerts
        self.thresholds = {
            'error_rate': 0.05,  # 5% error rate
            'response_time_p95': 30.0,  # 30 seconds
            'queue_size': 1000,  # Max queue size
            'memory_usage': 0.9  # 90% memory usage
        }
        
        # Start monitoring thread
        self._running = False
        self._monitor_thread = None
        
    def start(self):
        """Start monitoring"""
        if self._running:
            return
            
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.start()
        self.logger.info("Pipeline monitor started")
        
    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        self.logger.info("Pipeline monitor stopped")
        
    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """Add a health check
        
        Args:
            name: Component name
            check_func: Function that returns True if healthy
        """
        self.health_checks[name] = check_func
        
    def record_pipeline_execution(self, pipeline_name: str, duration_seconds: float,
                                 success: bool, error: Optional[str] = None):
        """Record pipeline execution metrics
        
        Args:
            pipeline_name: Pipeline name
            duration_seconds: Execution duration
            success: Whether execution succeeded
            error: Optional error message
        """
        labels = {'pipeline': pipeline_name, 'status': 'success' if success else 'failure'}
        
        # Record metrics
        self.metrics.increment_counter('pipeline_executions_total', labels=labels)
        self.metrics.record_time('pipeline_duration_seconds', duration_seconds, labels=labels)
        
        if not success:
            self.metrics.increment_counter('pipeline_errors_total', labels={'pipeline': pipeline_name})
            
            # Raise alert for pipeline failure
            self.alerts.raise_alert(Alert(
                alert_id=f"pipeline_failure_{pipeline_name}_{time.time()}",
                level=AlertLevel.ERROR,
                message=f"Pipeline '{pipeline_name}' failed: {error}",
                source="pipeline",
                metadata={'pipeline': pipeline_name, 'error': error}
            ))
            
    def record_api_request(self, endpoint: str, method: str, duration_seconds: float,
                          status_code: int):
        """Record API request metrics
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration_seconds: Request duration
            status_code: HTTP status code
        """
        labels = {
            'endpoint': endpoint,
            'method': method,
            'status': str(status_code)
        }
        
        self.metrics.increment_counter('api_requests_total', labels=labels)
        self.metrics.record_time('api_request_duration_seconds', duration_seconds, labels=labels)
        
        if status_code >= 500:
            self.metrics.increment_counter('api_errors_total', labels=labels)
            
    def get_health_status(self) -> HealthStatus:
        """Get current health status"""
        return self._health_status
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                # Run health checks
                self._check_health()
                
                # Check thresholds
                self._check_thresholds()
                
                # Clean old metrics
                self.metrics.clear_old_metrics()
                
            except Exception as e:
                self.logger.error(f"Monitor error: {str(e)}")
                
            time.sleep(30)  # Check every 30 seconds
            
    def _check_health(self):
        """Run health checks"""
        components = {}
        issues = []
        
        for name, check_func in self.health_checks.items():
            try:
                healthy = check_func()
                components[name] = "healthy" if healthy else "unhealthy"
                if not healthy:
                    issues.append(f"{name} is unhealthy")
            except Exception as e:
                components[name] = "error"
                issues.append(f"{name} check failed: {str(e)}")
                
        # Determine overall status
        if not issues:
            status = "healthy"
        elif len(issues) < len(components) / 2:
            status = "degraded"
        else:
            status = "unhealthy"
            
        self._health_status = HealthStatus(
            status=status,
            components=components,
            last_check=datetime.now(),
            issues=issues
        )
        
        # Raise alert if unhealthy
        if status == "unhealthy":
            self.alerts.raise_alert(Alert(
                alert_id="system_unhealthy",
                level=AlertLevel.CRITICAL,
                message="System health check failed",
                metadata={'issues': issues}
            ))
        elif status == "healthy":
            self.alerts.resolve_alert("system_unhealthy")
            
    def _check_thresholds(self):
        """Check metric thresholds"""
        summary = self.metrics.get_metrics_summary()
        
        # Check error rate
        total_requests = summary.get('api_requests_total', {}).get('total', 0)
        total_errors = summary.get('api_errors_total', {}).get('total', 0)
        
        if total_requests > 100:  # Only check after sufficient requests
            error_rate = total_errors / total_requests
            if error_rate > self.thresholds['error_rate']:
                self.alerts.raise_alert(Alert(
                    alert_id="high_error_rate",
                    level=AlertLevel.WARNING,
                    message=f"High error rate: {error_rate:.1%}",
                    metadata={'error_rate': error_rate}
                ))
                
        # Check response times
        response_times = summary.get('api_request_duration_seconds', {})
        if 'p95' in response_times:
            p95 = response_times['p95']
            if p95 > self.thresholds['response_time_p95']:
                self.alerts.raise_alert(Alert(
                    alert_id="slow_response_times",
                    level=AlertLevel.WARNING,
                    message=f"Slow response times: p95={p95:.1f}s",
                    metadata={'p95': p95}
                ))