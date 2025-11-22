import logging
from typing import List, Dict, Any, Set
from functools import wraps
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Permission(Enum):
    """User permissions"""
    PROJECT_CREATE = "project_create"
    PROJECT_READ = "project_read"
    PROJECT_UPDATE = "project_update"
    PROJECT_DELETE = "project_delete"
    MODEL_TRAIN = "model_train"
    MODEL_DEPLOY = "model_deploy"
    DATA_UPLOAD = "data_upload"
    DATA_DELETE = "data_delete"
    USER_MANAGE = "user_manage"
    SYSTEM_ADMIN = "system_admin"

class Role:
    """User role with permissions"""
    
    def __init__(self, name: str, permissions: Set[Permission]):
        self.name = name
        self.permissions = permissions
    
    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions

# Predefined roles
ROLES = {
    "admin": Role("admin", {
        Permission.PROJECT_CREATE, Permission.PROJECT_READ, Permission.PROJECT_UPDATE, Permission.PROJECT_DELETE,
        Permission.MODEL_TRAIN, Permission.MODEL_DEPLOY, Permission.DATA_UPLOAD, Permission.DATA_DELETE,
        Permission.USER_MANAGE, Permission.SYSTEM_ADMIN
    }),
    "researcher": Role("researcher", {
        Permission.PROJECT_CREATE, Permission.PROJECT_READ, Permission.PROJECT_UPDATE,
        Permission.MODEL_TRAIN, Permission.DATA_UPLOAD
    }),
    "annotator": Role("annotator", {
        Permission.PROJECT_READ, Permission.DATA_UPLOAD
    }),
    "viewer": Role("viewer", {
        Permission.PROJECT_READ
    })
}

class SecurityManager:
    """Manage security and access control"""
    
    def __init__(self, user_manager):
        self.user_manager = user_manager
    
    def get_user_role(self, user) -> Role:
        """Get user's role"""
        return ROLES.get(user.role, ROLES["viewer"])
    
    def check_permission(self, user, permission: Permission) -> bool:
        """Check if user has specific permission"""
        role = self.get_user_role(user)
        return role.has_permission(permission)
    
    def authorize_project_access(self, user, project_id: str, permission: Permission) -> bool:
        """Authorize project-specific access"""
        # Check basic permission
        if not self.check_permission(user, permission):
            return False
        
        # For project-specific permissions, verify user owns the project or is admin
        if user.role == "admin":
            return True
        
        # Verify user owns the project
        project = self.project_manager.get_project(project_id, user.id)
        return project is not None

def require_permission(permission: Permission):
    """Decorator to require specific permission"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, user, *args, **kwargs):
            if not self.security_manager.check_permission(user, permission):
                logger.warning(f"Permission denied for {user.username}: {permission}")
                return {"status": "error", "message": "Insufficient permissions"}
            return func(self, user, *args, **kwargs)
        return wrapper
    return decorator

def require_project_access(permission: Permission):
    """Decorator to require project access"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, user, project_id: str, *args, **kwargs):
            if not self.security_manager.authorize_project_access(user, project_id, permission):
                logger.warning(f"Project access denied for {user.username}: {project_id}")
                return {"status": "error", "message": "Access denied"}
            return func(self, user, project_id, *args, **kwargs)
        return wrapper
    return decorator

class AuditLogger:
    """Log security and user actions for audit purposes"""
    
    def __init__(self, db_path: str = "data/audit.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize audit database"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                details TEXT,
                ip_address TEXT,
                user_agent TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_action(self, user_id: str, action: str, resource_type: str = None,
                  resource_id: str = None, details: str = None, ip_address: str = None,
                  user_agent: str = None):
        """Log user action for audit purposes"""
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                """INSERT INTO audit_logs (user_id, action, resource_type, resource_id, details, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, action, resource_type, resource_id, details, ip_address, user_agent)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to log audit action: {e}")