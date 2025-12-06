#!/usr/bin/env python3
"""
Complete production launcher for Responsive Fine-Tuner
"""

import os
import sys
import argparse
import yaml
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deployment.production import ProductionDeployer
from monitoring.monitor import SystemMonitor
from maintenance.backup import BackupManager
from frontend.enterprise_interface import EnterpriseRFTInterface

class ProductionRFTManager:
    """Complete production management system"""
    
    def __init__(self, config_path: str = "config/production.yaml"):
        self.config = self._load_config(config_path)
        self.deployer = ProductionDeployer(config_path)
        self.monitor = SystemMonitor(self.config.get('monitoring', {}))
        self.backup_mgr = BackupManager(self.config.get('backup', {}))
        self.app_interface = None
        
        print("🏭 Production RFT Manager initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load production configuration"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def start_production_system(self):
        """Start the complete production system"""
        print("=" * 60)
        print("🚀 Starting Production Responsive Fine-Tuner")
        print("=" * 60)
        
        try:
            # 1. Deploy application
            print("\n1️⃣  Deploying application...")
            deployment_result = self.deployer.deploy_to_docker()
            
            if deployment_result['status'] != 'success':
                print(f"❌ Deployment failed: {deployment_result.get('error')}")
                return False
            
            print(f"✅ Application deployed on port {self.config.get('port', 7860)}")
            
            # 2. Start monitoring
            print("\n2️⃣  Starting system monitoring...")
            self.monitor.start_monitoring(interval=60)
            print("✅ System monitoring active")
            
            # 3. Schedule backups
            print("\n3️⃣  Configuring automated backups...")
            self.backup_mgr.schedule_backups()
            print("✅ Backup system configured")
            
            # 4. Initialize application interface
            print("\n4️⃣  Initializing application interface...")
            self.app_interface = EnterpriseRFTInterface(self.config.get('app_config'))
            
            # 5. Display system information
            print("\n" + "=" * 60)
            print("🏭 PRODUCTION SYSTEM READY")
            print("=" * 60)
            print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🌐 Application: http://localhost:{self.config.get('port', 7860)}")
            print(f"📊 Monitoring: Active (60s intervals)")
            print(f"💾 Backups: {self.config.get('backup', {}).get('schedule', 'daily')}")
            print(f"👥 Mode: {'Enterprise' if self.config.get('enterprise', False) else 'Standard'}")
            print("=" * 60)
            print("\nPress Ctrl+C to shutdown the system")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start production system: {e}")
            return False
    
    def stop_production_system(self):
        """Gracefully shutdown the production system"""
        print("\n" + "=" * 60)
        print("🛑 Shutting down production system...")
        print("=" * 60)
        
        try:
            # Stop monitoring
            self.monitor.stop_monitoring()
            print("✅ Monitoring stopped")
            
            # Create final backup
            print("Creating final backup...")
            backup_result = self.backup_mgr.create_backup()
            if backup_result['status'] == 'success':
                print(f"✅ Final backup created: {backup_result['name']}")
            
            # Display final status
            print("\n📊 Final System Status:")
            print(f"   - Total deployments: {len(self.deployer.deployment_history)}")
            print(f"   - Alerts triggered: {len(self.monitor.alerts)}")
            print(f"   - Last backup: {backup_result.get('name', 'N/A')}")
            print(f"   - Shutdown time: {datetime.now().strftime('%H:%M:%S')}")
            
            print("\n👋 Production system shutdown complete")
            
        except Exception as e:
            print(f"❌ Error during shutdown: {e}")
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            'deployment': self.deployer.get_deployment_status(),
            'monitoring': self.monitor.get_metrics_summary(),
            'backup': {
                'backup_dir': self.backup_mgr.backup_dir,
                'schedule': self.backup_mgr.backup_schedule
            },
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main production launcher"""
    parser = argparse.ArgumentParser(description="Production Responsive Fine-Tuner")
    parser.add_argument("--config", default="config/production.yaml", help="Production config file")
    parser.add_argument("--deploy-only", action="store_true", help="Only deploy, don't run interface")
    parser.add_argument("--monitor-only", action="store_true", help="Only run monitoring")
    parser.add_argument("--status", action="store_true", help="Check system status")
    
    args = parser.parse_args()
    
    try:
        manager = ProductionRFTManager(args.config)
        
        if args.status:
            status = manager.get_system_status()
            print(json.dumps(status, indent=2))
            return
        
        if args.monitor_only:
            print("Starting monitoring only...")
            manager.monitor.start_monitoring()
            return
        
        if args.deploy_only:
            print("Deploying only...")
            manager.deployer.deploy_to_docker()
            return
        
        # Start complete system
        if manager.start_production_system():
            try:
                # Keep the system running
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.stop_production_system()
    
    except Exception as e:
        print(f"❌ Production system error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()