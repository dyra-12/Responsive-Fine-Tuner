import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

from backend.auth import UserManager, User, JWTManager
from backend.project_manager import ProjectManager, Project
from backend.advanced_trainer import AdvancedModelManager
from backend.analytics import ModelAnalytics, DataQualityAnalyzer
from backend.optimizations import MemoryOptimizer, BackgroundTrainer
from backend.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnterpriseRFTApplication:
    """Enterprise version with multi-user support and project management"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.user_manager = UserManager()
        self.jwt_manager = JWTManager(self.user_manager.secret_key)
        self.project_manager = ProjectManager()
        
        # User session state
        self.active_sessions: Dict[str, User] = {}
        self.user_applications: Dict[str, UserApplication] = {}
        
        # System components (initialized per user)
        self.memory_optimizer = MemoryOptimizer()
        # Available labels derived from config model settings
        try:
            num_labels = getattr(self.config_manager.model, 'num_labels', 2)
        except Exception:
            num_labels = 2
        self.available_labels = [f"Class {i}" for i in range(num_labels)]
        
        logger.info("Enterprise RFT Application initialized")
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return session token"""
        user = self.user_manager.authenticate_user(username, password)
        if user:
            token = self.jwt_manager.create_token(user)
            self.active_sessions[token] = user
            self._initialize_user_application(user)
            
            logger.info(f"User authenticated: {username}")
            return {
                'token': token,
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role
                }
            }
        return None
    
    def validate_token(self, token: str) -> Optional[User]:
        """Validate JWT token and return user"""
        if token in self.active_sessions:
            return self.active_sessions[token]
        
        payload = self.jwt_manager.validate_token(token)
        if payload:
            user = self.user_manager.get_user_by_username(payload['username'])
            if user:
                self.active_sessions[token] = user
                self._initialize_user_application(user)
                return user
        return None
    
    def logout_user(self, token: str):
        """Logout user and clear session"""
        if token in self.active_sessions:
            user = self.active_sessions[token]
            self._cleanup_user_application(user)
            del self.active_sessions[token]
            logger.info(f"User logged out: {user.username}")
    
    def _initialize_user_application(self, user: User):
        """Initialize application components for user"""
        if user.id not in self.user_applications:
            self.user_applications[user.id] = UserApplication(
                user_id=user.id,
                config=self.config_manager
            )
            logger.info(f"Initialized application for user: {user.username}")
    
    def _cleanup_user_application(self, user: User):
        """Clean up user application resources"""
        if user.id in self.user_applications:
            # Clear memory and resources
            self.user_applications[user.id].cleanup()
            del self.user_applications[user.id]
            logger.info(f"Cleaned up application for user: {user.username}")
    
    def get_user_application(self, user_id: str) -> Optional['UserApplication']:
        """Get user's application instance"""
        return self.user_applications.get(user_id)
    
    # Project management methods
    def create_user_project(self, user: User, name: str, description: str = "", 
                          config: Dict[str, Any] = None) -> Optional[Project]:
        """Create a new project for user"""
        return self.project_manager.create_project(user.id, name, description, config)
    
    def get_user_projects(self, user: User) -> List[Project]:
        """Get all projects for user"""
        return self.project_manager.list_user_projects(user.id)
    
    def get_user_project(self, user: User, project_id: str) -> Optional[Project]:
        """Get specific project for user"""
        return self.project_manager.get_project(project_id, user.id)
    
    def switch_user_project(self, user: User, project_id: str) -> bool:
        """Switch user's active project"""
        project = self.get_user_project(user, project_id)
        if project and user.id in self.user_applications:
            self.user_applications[user.id].set_active_project(project)
            return True
        return False

class UserApplication:
    """Individual user application instance with isolated state"""
    
    def __init__(self, user_id: str, config):
        self.user_id = user_id
        self.config = config
        self.active_project: Optional[Project] = None
        
        # Core components
        self.model_manager: Optional[AdvancedModelManager] = None
        self.analytics = ModelAnalytics()
        self.data_analyzer = DataQualityAnalyzer()
        self.background_trainer: Optional[BackgroundTrainer] = None
        
        # User state
        self.current_data = None
        self.train_data = None
        self.test_data = None
        self.labeling_history = []
        self.performance_history = []
        
        logger.info(f"User application initialized for user: {user_id}")
    
    def set_active_project(self, project: Project):
        """Set active project and load project data"""
        self.active_project = project
        
        # Initialize model manager with project-specific settings
        project_config = project.config.get('model_config', {})
        self.model_manager = AdvancedModelManager(self.config)
        
        # Load project model if exists
        active_model = self.project_manager.get_active_model(project.id)
        if active_model:
            self._load_project_model(active_model)
        
        logger.info(f"Active project set: {project.name}")
    
    def _load_project_model(self, model_info: Dict[str, Any]):
        """Load project-specific model"""
        try:
            model_path = model_info.get('model_path')
            if model_path and os.path.exists(model_path):
                # Implementation for loading saved model
                logger.info(f"Loaded project model: {model_info['model_name']}")
        except Exception as e:
            logger.error(f"Failed to load project model: {e}")
    
    def save_project_state(self):
        """Save current project state"""
        if not self.active_project or not self.model_manager:
            return
        
        try:
            # Save model
            model_path = f"models/user_{self.user_id}/project_{self.active_project.id}"
            os.makedirs(model_path, exist_ok=True)
            
            # Save model and metrics
            metrics = {
                'performance_history': self.performance_history,
                'labeling_history': len(self.labeling_history),
                'last_updated': datetime.now().isoformat()
            }
            
            self.model_manager.save_model(model_path)
            self.project_manager.save_project_model(
                self.active_project.id,
                f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                model_path,
                metrics
            )
            
            logger.info(f"Project state saved: {self.active_project.name}")
        except Exception as e:
            logger.error(f"Failed to save project state: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.model_manager:
            self.save_project_state()
        
        # Clear memory
        self.current_data = None
        self.train_data = None
        self.test_data = None
        self.labeling_history.clear()
        self.performance_history.clear()
        
        if self.background_trainer:
            self.background_trainer = None
        
        logger.info(f"User application cleaned up: {self.user_id}")
    
    # Delegate methods to model manager
    def process_uploaded_files(self, files: List[str]) -> Dict[str, Any]:
        """Process uploaded files for current project"""
        if not self.model_manager:
            return {"status": "error", "message": "No active project"}
        
        # Implementation from RFTApplication
        pass
    
    def get_next_document(self) -> Tuple[Optional[str], Dict[str, Any]]:
        """Get next document for labeling"""
        if not self.model_manager:
            return None, {"status": "error", "message": "No active project"}
        
        # Implementation from RFTApplication
        pass
    
    def submit_feedback(self, document: str, user_feedback: str, correct_label: Optional[str] = None) -> Dict[str, Any]:
        """Submit feedback for current project"""
        if not self.model_manager:
            return {"status": "error", "message": "No active project"}
        
        # Implementation from RFTApplication
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for current project"""
        if not self.model_manager:
            return {"error": "No active project"}
        
        # Implementation from RFTApplication
        pass