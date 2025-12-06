import gradio as gr
import os
import sys
from typing import Dict, Any, Tuple, List, Optional

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from frontend.app_core import RFTApplication
from frontend.components.data_upload import create_data_upload_interface
from frontend.components.labeling import create_labeling_interface
from frontend.components.performance import create_performance_interface

class RFTInterface:
    """Main Gradio interface for Responsive Fine-Tuner"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.app = RFTApplication(config_path)
        self.interface = None
        
    def setup_interface(self):
        """Setup the complete Gradio interface"""
        
        # Some Gradio versions do not accept a `theme` kwarg on BlockContext.
        # Try with `theme` first and fall back to a no-theme Blocks if needed.
        try:
            with gr.Blocks(
                title="Responsive Fine-Tuner",
                theme=gr.themes.Soft(),
                css=self._get_custom_css()
            ) as interface:
                pass  # placeholder to establish the context in the try-block
        except TypeError:
            # Older Gradio versions may not accept `theme` or `css` kwargs.
            # Fall back to creating a Blocks with only the title.
            with gr.Blocks(title="Responsive Fine-Tuner") as interface:
                pass

        # Re-open the established context to populate the interface elements
        with interface:
            
            # Header
            gr.Markdown("# 🎯 Responsive Fine-Tuner")
            gr.Markdown("### Interactive Model Fine-Tuning for Domain Experts")
            gr.Markdown("Upload your documents, provide feedback, and watch your model improve in real-time!")
            
            # Main tabs
            with gr.Tabs() as tabs:
                # Data Upload Tab
                with gr.Tab("📁 Data Upload"):
                    data_interface = create_data_upload_interface(
                        self.app, 
                        self._process_files_wrapper
                    )
                
                # Interactive Labeling Tab
                with gr.Tab("🎯 Interactive Labeling"):
                    labeling_interface = create_labeling_interface(
                        self.app,
                        self._get_next_document_wrapper,
                        self._submit_feedback_wrapper
                    )
                
                # Performance Monitoring Tab
                with gr.Tab("📊 Performance"):
                    performance_interface = create_performance_interface(
                        self.app,
                        self._get_metrics_wrapper
                    )
                
                # Settings Tab
                with gr.Tab("⚙️ Settings"):
                    settings_interface = self._create_settings_interface()
            
            # Footer
            gr.Markdown("---")
            gr.Markdown(
                "**Responsive Fine-Tuner** • "
                "Human-in-the-Loop Machine Learning • "
                "Built with Gradio & Hugging Face Transformers"
            )
        
        self.interface = interface
        return interface
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for styling"""
        return """
        .gradio-container {
            max-width: 1200px !important;
        }
        .file-upload {
            min-height: 200px;
        }
        .progress-bar {
            margin: 10px 0;
        }
        """
    
    def _create_settings_interface(self):
        """Create settings interface"""
        with gr.Blocks() as interface:
            gr.Markdown("## ⚙️ Application Settings")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model Settings")
                    
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "distilbert-base-uncased",
                            "bert-base-uncased", 
                            "roberta-base"
                        ],
                        value="distilbert-base-uncased",
                        label="Base Model"
                    )
                    
                    learning_rate = gr.Slider(
                        minimum=1e-6,
                        maximum=1e-3,
                        value=1e-4,
                        step=1e-6,
                        label="Learning Rate"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        label="Batch Size"
                    )
                
                with gr.Column():
                    gr.Markdown("### Application Settings")
                    
                    update_interval = gr.Slider(
                        minimum=100,
                        maximum=5000,
                        value=1000,
                        step=100,
                        label="UI Update Interval (ms)"
                    )
                    
                    auto_retrain = gr.Checkbox(
                        value=True,
                        label="Auto-retrain after feedback batch"
                    )
                    
                    with gr.Row():
                        save_btn = gr.Button("Save Settings", variant="primary")
                        reset_btn = gr.Button("Reset Application", variant="secondary")
            
            # Event handlers
            save_btn.click(
                fn=self._save_settings,
                inputs=[model_dropdown, learning_rate, batch_size, update_interval, auto_retrain],
                outputs=[]
            )
            
            reset_btn.click(
                fn=self._reset_application,
                outputs=[]
            )
        
        return interface
    
    # Wrapper methods for Gradio callbacks
    def _process_files_wrapper(self, files, test_split, max_length):
        """Wrapper for file processing callback"""
        try:
            # Update config
            self.app.config.data.test_split = test_split
            self.app.config.model.max_length = max_length
            
            # Process files
            file_paths = [file.name for file in files] if files else []
            result = self.app.process_uploaded_files(file_paths)
            
            # Format responses
            from frontend.components.data_upload import create_upload_status_display, create_dataset_info_df
            status_display = create_upload_status_display(result)
            dataset_info = create_dataset_info_df(result)
            
            return status_display, dataset_info
            
        except Exception as e:
            return {"status": "error", "message": str(e)}, [["Error", str(e)]]
    
    def _get_next_document_wrapper(self):
        """Wrapper for getting next document"""
        try:
            document, doc_data = self.app.get_next_document()
            
            if document is None:
                # No more documents or error
                error_msg = doc_data.get("message", "No documents available")
                return (
                    error_msg,  # document_display
                    {"Error": 1.0},  # model_prediction
                    self._create_error_plot(),  # confidence_plot
                    "Labeling Complete",  # progress_text
                    100,  # progress_bar
                    None  # current_doc_data
                )
            
            # Create display elements
            from frontend.components.labeling import (
                create_prediction_display, 
                create_progress_display
            )
            
            prediction_display, confidence_plot = create_prediction_display(doc_data["prediction"])
            progress_text, progress_percentage = create_progress_display(
                doc_data["document_index"], 
                doc_data["total_documents"]
            )
            
            return (
                document,
                prediction_display,
                confidence_plot,
                progress_text,
                progress_percentage,
                doc_data
            )
            
        except Exception as e:
            error_msg = f"Error loading document: {str(e)}"
            return (
                error_msg,
                {"Error": 1.0},
                self._create_error_plot(),
                "Error",
                0,
                None
            )
    
    def _submit_feedback_wrapper(self, document, user_feedback, correct_label):
        """Wrapper for feedback submission"""
        try:
            result = self.app.submit_feedback(document, user_feedback, correct_label)
            
            # Update progress display
            from frontend.components.labeling import create_progress_display
            progress_text, progress_percentage = create_progress_display(
                result["current_index"],
                result["total_documents"]
            )
            
            return progress_text, progress_percentage, None
            
        except Exception as e:
            return f"Error: {str(e)}", 0, None
    
    def _get_metrics_wrapper(self):
        """Wrapper for getting performance metrics"""
        try:
            metrics = self.app.get_performance_metrics()
            
            from frontend.components.performance import (
                create_current_metrics_display,
                create_performance_plots
            )
            
            current_metrics = create_current_metrics_display(metrics)
            model_info = metrics.get("model_info", {})
            plots = create_performance_plots(metrics)
            
            return current_metrics, model_info, *plots
            
        except Exception as e:
            error_metrics = {"error": str(e)}
            error_plot = self._create_error_plot()
            return error_metrics, {}, error_plot, error_plot, error_plot
    
    def _save_settings(self, model_name, learning_rate, batch_size, update_interval, auto_retrain):
        """Save application settings"""
        # This would update the config and potentially reload the model
        # For now, just print the values
        print(f"Settings saved: {model_name}, lr={learning_rate}, batch={batch_size}")
    
    def _reset_application(self):
        """Reset application state"""
        self.app.reset_application()
        return gr.update()
    
    def _create_error_plot(self):
        """Create an error plot for display"""
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(
            text="Error",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Error Display",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        if self.interface is None:
            self.setup_interface()
        # Some Gradio versions accept a `queue`/`enable_queue` kwarg on launch,
        # others require calling `.queue()` on the Blocks object before launch.
        # Normalize by handling both possibilities here.
        if 'enable_queue' in kwargs:
            try:
                if kwargs.pop('enable_queue'):
                    self.interface = self.interface.queue()
            except Exception:
                # If `.queue()` is not available, just remove the flag and continue
                kwargs.pop('enable_queue', None)

        if 'queue' in kwargs:
            try:
                if kwargs.pop('queue'):
                    self.interface = self.interface.queue()
            except Exception:
                kwargs.pop('queue', None)

        return self.interface.launch(**kwargs)