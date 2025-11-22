import gradio as gr
import os
import sys
from typing import Dict, Any, Optional
import jwt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.enterprise_app import EnterpriseRFTApplication
from backend.auth import User
from frontend.components.data_upload import create_data_upload_interface
from frontend.components.labeling import create_labeling_interface
from frontend.components.performance import create_performance_interface
from frontend.components.analytics import create_analytics_interface

class EnterpriseRFTInterface:
    """Enterprise Gradio interface with authentication and multi-user support"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.app = EnterpriseRFTApplication(config_path)
        self.interface = None
        self.current_user: Optional[User] = None
        
    def setup_interface(self):
        """Setup the complete enterprise Gradio interface"""
        
        with gr.Blocks(
            title="Enterprise Responsive Fine-Tuner",
            theme=gr.themes.Soft(),
            css=self._get_enterprise_css()
        ) as interface:
            
            # Authentication state
            auth_token = gr.State()
            current_user = gr.State()
            
            # Main interface with authentication
            with gr.Column(visible=True) as login_section:
                gr.Markdown("# 🔐 Enterprise Responsive Fine-Tuner")
                gr.Markdown("### Secure Multi-User Model Fine-Tuning Platform")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Column():
                            gr.Markdown("### Login")
                            username = gr.Textbox(label="Username", placeholder="Enter your username")
                            password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
                            login_btn = gr.Button("Login", variant="primary")
                            login_status = gr.JSON(label="Login Status", visible=False)
                    
                    with gr.Column(scale=1):
                        with gr.Column():
                            gr.Markdown("### Register")
                            reg_username = gr.Textbox(label="Username", placeholder="Choose a username")
                            reg_email = gr.Textbox(label="Email", placeholder="Your email address")
                            reg_password = gr.Textbox(label="Password", type="password", placeholder="Choose a password")
                            reg_confirm = gr.Textbox(label="Confirm Password", type="password", placeholder="Confirm password")
                            register_btn = gr.Button("Register", variant="secondary")
                            register_status = gr.JSON(label="Registration Status", visible=False)
            
            # Main application (initially hidden)
            with gr.Column(visible=False) as main_app:
                # Header with user info and logout
                with gr.Row():
                    gr.Markdown("# 🏢 Enterprise Responsive Fine-Tuner")
                    with gr.Column(scale=1, min_width=200):
                        user_info = gr.JSON(label="User Info", show_label=False)
                        logout_btn = gr.Button("Logout", variant="stop", size="sm")
                
                # Project selection
                with gr.Row():
                    with gr.Column(scale=1):
                        project_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select Project",
                            interactive=True
                        )
                        new_project_name = gr.Textbox(label="New Project Name", placeholder="Enter project name")
                        new_project_desc = gr.Textbox(label="Description", placeholder="Project description")
                        create_project_btn = gr.Button("Create Project", variant="primary")
                    
                    with gr.Column(scale=2):
                        project_info = gr.JSON(label="Project Information")
                
                # Main tabs (only visible when project is selected)
                with gr.Tabs(visible=False) as main_tabs:
                    # Existing tabs from main_app.py
                    with gr.TabItem("📁 Data Upload"):
                        data_interface = create_data_upload_interface(
                            self.app, self._process_files_wrapper
                        )
                    
                    with gr.TabItem("🎯 Interactive Labeling"):
                        labeling_interface = create_labeling_interface(
                            self.app, self._get_next_document_wrapper, self._submit_feedback_wrapper
                        )
                    
                    with gr.TabItem("📊 Performance"):
                        performance_interface = create_performance_interface(
                            self.app, self._get_metrics_wrapper
                        )
                    
                    with gr.TabItem("📈 Analytics"):
                        analytics_interface = create_analytics_interface(
                            self.app, self._get_analytics_wrapper
                        )
                    
                    with gr.TabItem("👥 User Management") as admin_tab:
                        admin_interface = self._create_admin_interface()
            
            # Event handlers for authentication
            login_btn.click(
                fn=self._login_user,
                inputs=[username, password],
                outputs=[login_status, auth_token, current_user, login_section, main_app]
            )
            
            register_btn.click(
                fn=self._register_user,
                inputs=[reg_username, reg_email, reg_password, reg_confirm],
                outputs=[register_status]
            )
            
            logout_btn.click(
                fn=self._logout_user,
                inputs=[auth_token],
                outputs=[auth_token, current_user, login_section, main_app, main_tabs]
            )
            
            # Project management events
            create_project_btn.click(
                fn=self._create_project,
                inputs=[current_user, new_project_name, new_project_desc],
                outputs=[project_dropdown, project_info]
            )
            
            project_dropdown.change(
                fn=self._switch_project,
                inputs=[current_user, project_dropdown],
                outputs=[project_info, main_tabs]
            )
        
        self.interface = interface
        return interface
    
    def _get_enterprise_css(self) -> str:
        """Get enterprise-specific CSS"""
        return """
        .gradio-container {
            max-width: 1400px !important;
        }
        .enterprise-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }
        .user-info-panel {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
        }
        """
    
    def _login_user(self, username: str, password: str):
        """Handle user login"""
        try:
            result = self.app.authenticate_user(username, password)
            if result:
                user_info = {
                    "username": result['user']['username'],
                    "email": result['user']['email'],
                    "role": result['user']['role'],
                    "login_time": "Now"
                }
                
                # Load user projects
                projects = self.app.get_user_projects(self.app.active_sessions[result['token']])
                project_choices = [f"{p.name} ({p.id})" for p in projects]
                
                return (
                    {"status": "success", "message": f"Welcome {username}!"},
                    result['token'],
                    result['user'],
                    gr.update(visible=False),  # Hide login
                    gr.update(visible=True)   # Show main app
                )
            else:
                return (
                    {"status": "error", "message": "Invalid credentials"},
                    None,
                    None,
                    gr.update(visible=True),
                    gr.update(visible=False)
                )
        except Exception as e:
            return (
                {"status": "error", "message": f"Login failed: {str(e)}"},
                None,
                None,
                gr.update(visible=True),
                gr.update(visible=False)
            )
    
    def _register_user(self, username: str, email: str, password: str, confirm: str):
        """Handle user registration"""
        try:
            if password != confirm:
                return {"status": "error", "message": "Passwords do not match"}
            
            if len(password) < 6:
                return {"status": "error", "message": "Password must be at least 6 characters"}
            
            user = self.app.user_manager.create_user(username, email, password)
            if user:
                return {"status": "success", "message": f"User {username} created successfully!"}
            else:
                return {"status": "error", "message": "Username or email already exists"}
        except Exception as e:
            return {"status": "error", "message": f"Registration failed: {str(e)}"}
    
    def _logout_user(self, token: str):
        """Handle user logout"""
        if token:
            self.app.logout_user(token)
        
        return (
            None,  # Clear auth token
            None,  # Clear current user
            gr.update(visible=True),   # Show login
            gr.update(visible=False),  # Hide main app
            gr.update(visible=False)   # Hide main tabs
        )
    
    def _create_project(self, user_data: Dict, name: str, description: str):
        """Create new project for user"""
        try:
            if not user_data:
                return [], {"error": "Not authenticated"}
            
            user = self.app.active_sessions.get(list(self.app.active_sessions.keys())[0])
            project = self.app.create_user_project(user, name, description)
            
            if project:
                # Refresh project list
                projects = self.app.get_user_projects(user)
                project_choices = [f"{p.name} ({p.id})" for p in projects]
                project_info = {
                    "name": project.name,
                    "description": project.description,
                    "created_at": project.created_at,
                    "status": "active"
                }
                
                return gr.update(choices=project_choices), project_info
            else:
                return [], {"error": "Failed to create project"}
        except Exception as e:
            return [], {"error": f"Project creation failed: {str(e)}"}
    
    def _switch_project(self, user_data: Dict, project_selection: str):
        """Switch active project"""
        try:
            if not user_data or not project_selection:
                return {"error": "No project selected"}, gr.update(visible=False)
            
            # Extract project ID from selection
            project_id = project_selection.split('(')[-1].rstrip(')')
            user = self.app.active_sessions.get(list(self.app.active_sessions.keys())[0])
            
            success = self.app.switch_user_project(user, project_id)
            if success:
                project = self.app.get_user_project(user, project_id)
                project_info = {
                    "name": project.name,
                    "description": project.description,
                    "status": project.status,
                    "models": len(self.app.project_manager.get_active_model(project_id) or [])
                }
                return project_info, gr.update(visible=True)
            else:
                return {"error": "Failed to switch project"}, gr.update(visible=False)
        except Exception as e:
            return {"error": f"Project switch failed: {str(e)}"}, gr.update(visible=False)
    
    def _create_admin_interface(self):
        """Create admin interface for user management"""
        with gr.Blocks() as interface:
            gr.Markdown("## 👥 User Management")
            gr.Markdown("Administrator panel for user and system management.")
            
            with gr.Tabs():
                with gr.TabItem("Users"):
                    users_table = gr.Dataframe(
                        headers=["ID", "Username", "Email", "Role", "Last Login"],
                        label="System Users"
                    )
                    refresh_users_btn = gr.Button("Refresh Users")
                
                with gr.TabItem("System Metrics"):
                    system_metrics = gr.JSON(label="System Performance")
                    refresh_metrics_btn = gr.Button("Refresh Metrics")
            
            # Event handlers
            refresh_users_btn.click(
                fn=self._get_system_users,
                outputs=[users_table]
            )
            
            refresh_metrics_btn.click(
                fn=self._get_system_metrics,
                outputs=[system_metrics]
            )
        
        return interface
    
    def _get_system_users(self):
        """Get system users for admin panel"""
        # This would query the database for all users
        # For now, return mock data
        return [
            ["1", "admin", "admin@company.com", "admin", "2024-01-15"],
            ["2", "researcher1", "res1@company.com", "researcher", "2024-01-14"],
            ["3", "annotator1", "ann1@company.com", "annotator", "2024-01-13"]
        ]
    
    def _get_system_metrics(self):
        """Get system performance metrics"""
        return {
            "active_users": len(self.app.active_sessions),
            "total_projects": "25",
            "system_memory": "4.2GB / 16GB",
            "active_training": "2 sessions",
            "uptime": "5 days, 3 hours"
        }
    
    # Wrapper methods for existing functionality
    def _process_files_wrapper(self, files, test_split, max_length):
        """Wrapper for file processing with user context"""
        try:
            # In production this should validate user and delegate to user application
            # For now return a standardized error/success placeholder
            if not files:
                return ({"status": "error", "message": "No files provided"}, [["Status", "No data"]])

            # Minimal simulated response
            result = {
                "status": "success",
                "message": "Files processed (simulated)",
                "document_count": len(files) if isinstance(files, (list, tuple)) else 1,
                "train_count": 1,
                "test_count": 0,
                "file_sources": [str(f) for f in (files if isinstance(files, (list, tuple)) else [files])]
            }

            return (result, [
                ["Total Documents", result["document_count"]],
                ["Training Samples", result["train_count"]],
                ["Test Samples", result["test_count"]],
                ["File Sources", len(result["file_sources"])],
                ["Status", "Ready for Labeling"]
            ])
        except Exception as e:
            return ({"status": "error", "message": str(e)}, [["Status", "Error"]])
    
    def _get_next_document_wrapper(self):
        """Wrapper for document retrieval with user context"""
        # Return placeholders expected by the labeling interface
        try:
            document = "No documents available"
            model_prediction = {"No Data": 1.0}
            confidence_plot = None
            progress_text = "No documents available"
            progress_percentage = 0
            current_doc_data = None

            return document, model_prediction, confidence_plot, progress_text, progress_percentage, current_doc_data
        except Exception:
            return "No documents", {"No Data": 1.0}, None, "No documents", 0, None

    def _submit_feedback_wrapper(self, document: str, user_feedback: str, correct_label: Optional[str] = None):
        """Handle feedback submission (placeholder)"""
        try:
            # Placeholder: record would be persisted in real implementation
            progress_text = "Feedback submitted (simulated)"
            progress_percentage = 0
            current_doc_data = None
            return progress_text, progress_percentage, current_doc_data
        except Exception as e:
            return f"Error: {e}", 0, None

    def _get_metrics_wrapper(self):
        """Return performance metrics placeholders"""
        try:
            metrics = {
                "current_accuracy": 0.0,
                "labeling_progress": 0.0,
                "labeled_count": 0,
                "total_count": 0,
                "feedback_count": 0,
                "training_sessions": 0,
                "performance_history": []
            }

            model_info = {"model_name": "none", "training_sessions": 0}
            accuracy_plot = None
            training_plot = None
            label_plot = None

            return metrics, model_info, accuracy_plot, training_plot, label_plot
        except Exception:
            return {"error": "failed to get metrics"}, {}, None, None, None

    def _get_analytics_wrapper(self):
        """Return analytics placeholders"""
        try:
            analytics_data = {
                "current_performance": {},
                "data_quality": {},
                "training_analytics": {},
                "timestamp": None
            }

            # Return placeholders matching outputs expected by analytics UI
            return (
                analytics_data.get("current_performance"),
                None, None,  # performance_plot, confidence_plot
                analytics_data.get("data_quality"),
                analytics_data.get("data_quality"),
                None, None,  # text_length_plot, label_distribution_plot
                analytics_data.get("training_analytics"),
                None, None   # learning_rate_plot, loss_trend_plot
            )
        except Exception:
            return {"error": "analytics error"}, None, None, None, None, None, None, None, None, None
    
    # ... other wrapper methods
    
    def launch(self, **kwargs):
        """Launch the enterprise interface"""
        if self.interface is None:
            self.setup_interface()
        
        return self.interface.launch(**kwargs)