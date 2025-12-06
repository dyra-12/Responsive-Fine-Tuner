import pytest
import sys
import os
import tempfile
import subprocess
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deployment.production import ProductionDeployer
from monitoring.monitor import SystemMonitor
from maintenance.backup import BackupManager

class TestPhase6:
    """Test suite for Phase 6 production features"""
    
    def setup_method(self):
        """Setup before each test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test configs
        self.deploy_config = {
            'port': 7860,
            'replicas': 2,
            'environment': 'test'
        }
        
        self.monitor_config = {
            'email_alerts': False,
            'slack_webhook': None
        }
        
        self.backup_config = {
            'backup_dir': os.path.join(self.temp_dir, 'backups'),
            'retention_days': 7,
            'schedule': 'daily'
        }
    
    def test_production_deployer(self):
        """Test production deployment system"""
        deployer = ProductionDeployer()
        
        # Mock subprocess calls for Docker
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "container_id_123"
            
            result = deployer.deploy_to_docker()
            
            assert result['type'] == 'docker'
            assert result['status'] == 'success'
            assert 'container_id' in result
        
        print("✓ Production deployer test passed")
    
    def test_system_monitor(self):
        """Test system monitoring"""
        monitor = SystemMonitor(self.monitor_config)
        
        # Test metrics collection
        metrics = monitor.collect_metrics()
        
        assert 'timestamp' in metrics
        assert 'system' in metrics
        assert 'application' in metrics
        assert 'model' in metrics
        
        # Test system metrics
        system_metrics = metrics['system']
        assert 'cpu_percent' in system_metrics
        assert 'memory_percent' in system_metrics
        
        print("✓ System monitor test passed")
    
    def test_alert_thresholds(self):
        """Test alert threshold system"""
        monitor = SystemMonitor(self.monitor_config)
        
        # Create test metrics that should trigger alerts
        test_metrics = {
            'timestamp': '2024-01-01T00:00:00',
            'system': {'cpu_percent': 95, 'memory_percent': 90},
            'application': {'response_time_ms': 6000, 'status': 'healthy'},
            'model': {'active_models': 1}
        }
        
        # Mock alert sending
        with patch.object(monitor, '_send_critical_alert') as mock_alert:
            monitor._check_thresholds(test_metrics)
            
            # Should trigger alerts for CPU and memory
            assert mock_alert.called
        
        print("✓ Alert thresholds test passed")
    
    def test_backup_manager(self):
        """Test backup and recovery system"""
        backup_mgr = BackupManager(self.backup_config)
        
        # Create test directory structure
        test_data_dir = os.path.join(self.temp_dir, 'data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Create test file
        test_file = os.path.join(test_data_dir, 'test.txt')
        with open(test_file, 'w') as f:
            f.write("Test data")
        
        # Mock actual backup operations
        with patch.object(backup_mgr, '_backup_data'), \
             patch.object(backup_mgr, '_backup_models'), \
             patch.object(backup_mgr, '_backup_config'), \
             patch.object(backup_mgr, '_backup_database'):
            
            result = backup_mgr.create_backup('full')
            
            assert result['status'] == 'success'
            assert result['type'] == 'full'
        
        print("✓ Backup manager test passed")
    
    def test_backup_cleanup(self):
        """Test backup retention cleanup"""
        backup_mgr = BackupManager(self.backup_config)
        
        # Create test backup files with different ages
        import time
        old_time = time.time() - (8 * 24 * 3600)  # 8 days ago
        
        # Create "old" backup file
        old_backup = os.path.join(backup_mgr.backup_dir, 'old_backup.tar.gz')
        with open(old_backup, 'w') as f:
            f.write("old backup")
        
        # Set modification time to 8 days ago
        os.utime(old_backup, (old_time, old_time))
        
        # Create "recent" backup file
        recent_backup = os.path.join(backup_mgr.backup_dir, 'recent_backup.tar.gz')
        with open(recent_backup, 'w') as f:
            f.write("recent backup")
        
        # Run cleanup
        backup_mgr._cleanup_old_backups()
        
        # Verify old backup was removed, recent backup remains
        assert not os.path.exists(old_backup)
        assert os.path.exists(recent_backup)
        
        print("✓ Backup cleanup test passed")
    
    def test_metrics_summary(self):
        """Test metrics summary generation"""
        monitor = SystemMonitor(self.monitor_config)
        
        # Add test metrics
        test_metric = {
            'timestamp': '2024-01-01T12:00:00',
            'system': {'cpu_percent': 50, 'memory_percent': 60},
            'application': {'response_time_ms': 100, 'status': 'healthy'},
            'model': {'active_models': 1}
        }
        
        monitor.metrics_history.append(test_metric)
        
        # Get summary
        summary = monitor.get_metrics_summary(hours=24)
        
        assert 'period_hours' in summary
        assert 'samples' in summary
        assert 'avg_cpu_percent' in summary
        
        print("✓ Metrics summary test passed")
    
    def test_deployment_status(self):
        """Test deployment status reporting"""
        deployer = ProductionDeployer()
        
        # Add test deployment to history
        deployer.deployment_history.append({
            'type': 'docker',
            'status': 'success',
            'timestamp': '2024-01-01T00:00:00'
        })
        
        status = deployer.get_deployment_status()
        
        assert 'active_deployments' in status
        assert 'last_deployment' in status
        assert 'system_status' in status
        
        print("✓ Deployment status test passed")

def run_phase6_tests():
    """Run all Phase 6 tests"""
    print("🚀 Running Phase 6 Tests...")
    print("=" * 50)
    
    test_suite = TestPhase6()
    
    try:
        test_suite.setup_method()
        test_suite.test_production_deployer()
        test_suite.test_system_monitor()
        test_suite.test_alert_thresholds()
        test_suite.test_backup_manager()
        test_suite.test_backup_cleanup()
        test_suite.test_metrics_summary()
        test_suite.test_deployment_status()
        
        print("=" * 50)
        print("🎉 ALL PHASE 6 TESTS PASSED!")
        print("✓ Production deployment system working")
        print("✓ System monitoring and alerting")
        print("✓ Backup and recovery management")
        print("✓ Metrics collection and analysis")
        print("✓ Alert threshold system")
        print("✓ Backup retention cleanup")
        print("✓ Deployment status reporting")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_phase6_tests()