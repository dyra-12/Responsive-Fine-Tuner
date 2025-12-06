import time
import threading
import psutil
import GPUtil
import requests
from typing import Dict, List, Any, Callable
from datetime import datetime
import logging
import smtplib
from email.mime.text import MIMEText
import slack_sdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitor:
    """Real-time system monitoring and alerting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = []
        self.alerts = []
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Alert thresholds
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'gpu_memory_percent': 90,
            'response_time_ms': 5000,
            'error_rate': 0.05
        }
    
    def start_monitoring(self, interval: int = 60):
        """Start monitoring in background thread"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds and trigger alerts
                self._check_thresholds(metrics)
                
                # Keep history limited
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': self._get_system_metrics(),
            'application': self._get_application_metrics(),
            'model': self._get_model_metrics()
        }
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3)
        }
        
        # GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Primary GPU
                metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                metrics['gpu_memory_used_gb'] = gpu.memoryUsed
                metrics['gpu_memory_total_gb'] = gpu.memoryTotal
                metrics['gpu_temperature'] = gpu.temperature
        except:
            pass  # GPU not available
        
        return metrics
    
    def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application-level metrics"""
        try:
            # Check application health
            response = requests.get('http://localhost:7860', timeout=5)
            response_time = response.elapsed.total_seconds() * 1000  # ms
            
            return {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time_ms': response_time,
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': 'unreachable',
                'error': str(e),
                'response_time_ms': 0
            }
    
    def _get_model_metrics(self) -> Dict[str, Any]:
        """Get model-specific metrics"""
        # This would query the model manager for metrics
        return {
            'active_models': 1,
            'training_sessions': 0,
            'inference_count': 0,
            'average_inference_time': 0
        }
    
    def _check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and trigger alerts"""
        system_metrics = metrics['system']
        
        # CPU check
        if system_metrics.get('cpu_percent', 0) > self.thresholds['cpu_percent']:
            self.trigger_alert(
                level='warning',
                metric='cpu_percent',
                value=system_metrics['cpu_percent'],
                threshold=self.thresholds['cpu_percent'],
                message=f"High CPU usage: {system_metrics['cpu_percent']}%"
            )
        
        # Memory check
        if system_metrics.get('memory_percent', 0) > self.thresholds['memory_percent']:
            self.trigger_alert(
                level='critical',
                metric='memory_percent',
                value=system_metrics['memory_percent'],
                threshold=self.thresholds['memory_percent'],
                message=f"High memory usage: {system_metrics['memory_percent']}%"
            )
        
        # Response time check
        app_metrics = metrics['application']
        if app_metrics.get('response_time_ms', 0) > self.thresholds['response_time_ms']:
            self.trigger_alert(
                level='warning',
                metric='response_time_ms',
                value=app_metrics['response_time_ms'],
                threshold=self.thresholds['response_time_ms'],
                message=f"Slow response time: {app_metrics['response_time_ms']:.0f}ms"
            )
    
    def trigger_alert(self, level: str, metric: str, value: float, 
                     threshold: float, message: str):
        """Trigger an alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'message': message
        }
        
        self.alerts.append(alert)
        logger.warning(f"Alert: {message}")
        
        # Send notifications based on level
        if level == 'critical':
            self._send_critical_alert(alert)
        elif level == 'warning':
            self._send_warning_alert(alert)
    
    def _send_critical_alert(self, alert: Dict[str, Any]):
        """Send critical alert notifications"""
        # Email notification
        if self.config.get('email_alerts'):
            self._send_email_alert(alert)
        
        # Slack notification
        if self.config.get('slack_webhook'):
            self._send_slack_alert(alert)
    
    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send email alert"""
        try:
            msg = MIMEText(
                f"CRITICAL ALERT: {alert['message']}\n"
                f"Time: {alert['timestamp']}\n"
                f"Metric: {alert['metric']}\n"
                f"Value: {alert['value']} (Threshold: {alert['threshold']})"
            )
            msg['Subject'] = f"[RFT] Critical Alert: {alert['metric']}"
            msg['From'] = self.config['email_from']
            msg['To'] = self.config['email_to']
            
            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['smtp_user'], self.config['smtp_password'])
                server.send_message(msg)
                
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        recent_metrics = [m for m in self.metrics_history 
                         if (datetime.now() - datetime.fromisoformat(m['timestamp'])).total_seconds() < hours * 3600]
        
        if not recent_metrics:
            return {"status": "no_data"}
        
        # Calculate averages
        cpu_values = [m['system'].get('cpu_percent', 0) for m in recent_metrics]
        memory_values = [m['system'].get('memory_percent', 0) for m in recent_metrics]
        response_times = [m['application'].get('response_time_ms', 0) for m in recent_metrics]
        
        return {
            'period_hours': hours,
            'samples': len(recent_metrics),
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'avg_response_time_ms': sum(response_times) / len(response_times),
            'alerts_last_24h': len([a for a in self.alerts if 
                                   (datetime.now() - datetime.fromisoformat(a['timestamp'])).total_seconds() < 24 * 3600]),
            'current_status': recent_metrics[-1]['application']['status']
        }