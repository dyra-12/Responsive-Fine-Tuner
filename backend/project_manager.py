import json
import secrets
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Project:
    id: str
    user_id: str
    name: str
    description: str
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    status: str

class ProjectManager:
    """Manage user projects with data isolation"""
    
    def __init__(self, db_path: str = "data/users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize project database"""
        with self._get_db_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    config TEXT,
                    status TEXT DEFAULT 'active',
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS project_models (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_path TEXT,
                    metrics TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS project_data (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT,
                    file_type TEXT,
                    processed_data TEXT,
                    stats TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES projects (id)
                )
            ''')
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def create_project(self, user_id: str, name: str, description: str = "", 
                      config: Dict[str, Any] = None) -> Optional[Project]:
        """Create new project for user"""
        try:
            project_id = secrets.token_urlsafe(16)
            config_json = json.dumps(config or {})
            
            with self._get_db_connection() as conn:
                conn.execute(
                    """INSERT INTO projects (id, user_id, name, description, config) 
                    VALUES (?, ?, ?, ?, ?)""",
                    (project_id, user_id, name, description, config_json)
                )
            
            logger.info(f"Project created: {name} for user {user_id}")
            return self.get_project(project_id, user_id)
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            return None
    
    def get_project(self, project_id: str, user_id: str) -> Optional[Project]:
        """Get project by ID (with user verification)"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM projects WHERE id = ? AND user_id = ?",
                    (project_id, user_id)
                )
                project_data = cursor.fetchone()
                
                if project_data:
                    return Project(
                        id=project_data['id'],
                        user_id=project_data['user_id'],
                        name=project_data['name'],
                        description=project_data['description'],
                        created_at=project_data['created_at'],
                        updated_at=project_data['updated_at'],
                        config=json.loads(project_data['config']),
                        status=project_data['status']
                    )
            return None
        except Exception as e:
            logger.error(f"Failed to get project: {e}")
            return None
    
    def list_user_projects(self, user_id: str) -> List[Project]:
        """List all projects for a user"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM projects WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,)
                )
                projects = []
                for row in cursor.fetchall():
                    projects.append(Project(
                        id=row['id'],
                        user_id=row['user_id'],
                        name=row['name'],
                        description=row['description'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        config=json.loads(row['config']),
                        status=row['status']
                    ))
                return projects
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []
    
    def update_project(self, project_id: str, user_id: str, **updates) -> Optional[Project]:
        """Update project details"""
        try:
            allowed_updates = {'name', 'description', 'config', 'status'}
            update_fields = {k: v for k, v in updates.items() if k in allowed_updates}
            
            if not update_fields:
                return self.get_project(project_id, user_id)
            
            set_clause = ", ".join(f"{field} = ?" for field in update_fields.keys())
            values = list(update_fields.values())
            
            # Handle config serialization
            if 'config' in update_fields:
                values[list(update_fields.keys()).index('config')] = json.dumps(updates['config'])
            
            values.extend([project_id, user_id])
            
            with self._get_db_connection() as conn:
                conn.execute(
                    f"UPDATE projects SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?",
                    values
                )
            
            logger.info(f"Project updated: {project_id}")
            return self.get_project(project_id, user_id)
        except Exception as e:
            logger.error(f"Failed to update project: {e}")
            return None
    
    def delete_project(self, project_id: str, user_id: str) -> bool:
        """Delete project and associated data"""
        try:
            with self._get_db_connection() as conn:
                # Delete associated data first
                conn.execute("DELETE FROM project_models WHERE project_id = ?", (project_id,))
                conn.execute("DELETE FROM project_data WHERE project_id = ?", (project_id,))
                # Delete project
                conn.execute("DELETE FROM projects WHERE id = ? AND user_id = ?", (project_id, user_id))
            
            logger.info(f"Project deleted: {project_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            return False
    
    def save_project_model(self, project_id: str, model_name: str, model_path: str, 
                          metrics: Dict[str, Any] = None) -> bool:
        """Save model information for a project"""
        try:
            model_id = secrets.token_urlsafe(16)
            metrics_json = json.dumps(metrics or {})
            
            # Deactivate other models for this project
            with self._get_db_connection() as conn:
                conn.execute(
                    "UPDATE project_models SET is_active = FALSE WHERE project_id = ?",
                    (project_id,)
                )
                
                # Save new model
                conn.execute(
                    """INSERT INTO project_models (id, project_id, model_name, model_path, metrics, is_active) 
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (model_id, project_id, model_name, model_path, metrics_json, True)
                )
            
            logger.info(f"Model saved for project {project_id}: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False
    
    def get_active_model(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get active model for project"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM project_models WHERE project_id = ? AND is_active = TRUE ORDER BY created_at DESC LIMIT 1",
                    (project_id,)
                )
                model_data = cursor.fetchone()
                
                if model_data:
                    return {
                        'id': model_data['id'],
                        'model_name': model_data['model_name'],
                        'model_path': model_data['model_path'],
                        'metrics': json.loads(model_data['metrics']),
                        'created_at': model_data['created_at']
                    }
            return None
        except Exception as e:
            logger.error(f"Failed to get active model: {e}")
            return None