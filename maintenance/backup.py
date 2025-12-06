import os
import shutil
import tarfile
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import schedule
import time
import threading
import boto3
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupManager:
    """Automated backup and recovery system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backup_dir = config.get('backup_dir', 'backups')
        self.retention_days = config.get('retention_days', 30)
        self.backup_schedule = config.get('schedule', 'daily')
        
        os.makedirs(self.backup_dir, exist_ok=True)
        
    def create_backup(self, backup_type: str = 'full') -> Dict[str, Any]:
        """Create a backup of the system"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"rft_backup_{timestamp}_{backup_type}"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            logger.info(f"Creating {backup_type} backup: {backup_name}")
            
            # Create backup structure
            os.makedirs(backup_path, exist_ok=True)
            
            # Backup data
            self._backup_data(backup_path)
            
            # Backup models
            self._backup_models(backup_path)
            
            # Backup configuration
            self._backup_config(backup_path)
            
            # Backup database
            self._backup_database(backup_path)
            
            # Create archive
            archive_path = f"{backup_path}.tar.gz"
            self._create_archive(backup_path, archive_path)
            
            # Cleanup temporary directory
            shutil.rmtree(backup_path)
            
            # Upload to cloud if configured
            if self.config.get('cloud_backup'):
                self._upload_to_cloud(archive_path)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            backup_info = {
                'name': backup_name,
                'type': backup_type,
                'path': archive_path,
                'size_mb': os.path.getsize(archive_path) / (1024 * 1024),
                'timestamp': timestamp,
                'status': 'success'
            }
            
            logger.info(f"Backup created successfully: {backup_name}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _backup_data(self, backup_path: str):
        """Backup user data"""
        data_src = 'data'
        data_dst = os.path.join(backup_path, 'data')
        
        if os.path.exists(data_src):
            shutil.copytree(data_src, data_dst)
            logger.info(f"Backed up data: {data_src} -> {data_dst}")
    
    def _backup_models(self, backup_path: str):
        """Backup trained models"""
        models_src = 'models'
        models_dst = os.path.join(backup_path, 'models')
        
        if os.path.exists(models_src):
            shutil.copytree(models_src, models_dst)
            logger.info(f"Backed up models: {models_src} -> {models_dst}")
    
    def _backup_config(self, backup_path: str):
        """Backup configuration files"""
        config_files = ['config/settings.yaml', 'config/enterprise.yaml']
        
        config_dst = os.path.join(backup_path, 'config')
        os.makedirs(config_dst, exist_ok=True)
        
        for config_file in config_files:
            if os.path.exists(config_file):
                shutil.copy2(config_file, config_dst)
                logger.info(f"Backed up config: {config_file}")
    
    def _backup_database(self, backup_path: str):
        """Backup SQLite databases"""
        db_files = ['data/users.db', 'data/audit.db']
        
        db_dst = os.path.join(backup_path, 'databases')
        os.makedirs(db_dst, exist_ok=True)
        
        for db_file in db_files:
            if os.path.exists(db_file):
                # Create a copy of the database
                backup_db = os.path.join(db_dst, os.path.basename(db_file))
                conn = sqlite3.connect(db_file)
                backup_conn = sqlite3.connect(backup_db)
                
                conn.backup(backup_conn)
                backup_conn.close()
                conn.close()
                
                logger.info(f"Backed up database: {db_file}")
    
    def _create_archive(self, source_dir: str, archive_path: str):
        """Create compressed archive"""
        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        
        logger.info(f"Created archive: {archive_path}")
    
    def _upload_to_cloud(self, file_path: str):
        """Upload backup to cloud storage"""
        provider = self.config['cloud_backup']['provider']
        
        if provider == 'aws':
            self._upload_to_s3(file_path)
        elif provider == 'gcp':
            self._upload_to_gcs(file_path)
        elif provider == 'azure':
            self._upload_to_azure(file_path)
    
    def _upload_to_s3(self, file_path: str):
        """Upload to AWS S3"""
        s3 = boto3.client('s3')
        bucket = self.config['cloud_backup']['bucket']
        key = f"backups/{os.path.basename(file_path)}"
        
        s3.upload_file(file_path, bucket, key)
        logger.info(f"Uploaded to S3: s3://{bucket}/{key}")
    
    def _cleanup_old_backups(self):
        """Remove backups older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        for filename in os.listdir(self.backup_dir):
            filepath = os.path.join(self.backup_dir, filename)
            
            if os.path.isfile(filepath):
                file_date = datetime.fromtimestamp(os.path.getctime(filepath))
                
                if file_date < cutoff_date:
                    os.remove(filepath)
                    logger.info(f"Removed old backup: {filename}")
    
    def restore_backup(self, backup_path: str, restore_type: str = 'full') -> Dict[str, Any]:
        """Restore system from backup"""
        try:
            logger.info(f"Restoring from backup: {backup_path}")
            
            # Extract backup
            extract_dir = os.path.join(self.backup_dir, 'restore_temp')
            os.makedirs(extract_dir, exist_ok=True)
            
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            
            backup_root = os.path.join(extract_dir, os.listdir(extract_dir)[0])
            
            # Restore based on type
            if restore_type in ['full', 'data']:
                self._restore_data(backup_root)
            
            if restore_type in ['full', 'models']:
                self._restore_models(backup_root)
            
            if restore_type in ['full', 'config']:
                self._restore_config(backup_root)
            
            if restore_type in ['full', 'database']:
                self._restore_database(backup_root)
            
            # Cleanup
            shutil.rmtree(extract_dir)
            
            logger.info("Restore completed successfully")
            return {'status': 'success', 'restored_type': restore_type}
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def schedule_backups(self):
        """Schedule automatic backups"""
        if self.backup_schedule == 'daily':
            schedule.every().day.at("02:00").do(self.create_backup)
        elif self.backup_schedule == 'weekly':
            schedule.every().monday.at("02:00").do(self.create_backup)
        elif self.backup_schedule == 'hourly':
            schedule.every().hour.do(self.create_backup)
        
        # Run scheduler in background
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info(f"Backup scheduler started: {self.backup_schedule}")
    
    def _run_scheduler(self):
        """Run the schedule in background"""
        while True:
            schedule.run_pending()
            time.sleep(60)