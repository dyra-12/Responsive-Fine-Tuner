import gradio as gr
import os
import sys

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data_manager import DataManager
from utils.model_manager import RFTModelManager
from utils.feedback_loop import FeedbackLoop

class ResponsiveFineTunerApp:
    def __init__(self):
        self.data_manager = DataManager()
        self.model_manager = None
        self.feedback_loop = FeedbackLoop()
        
        # App state
        self.current_pool_df = None
        self.current_test_df = None
        self.labeling_sample_df = None
        
        self._create_interface()
    
    def _create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Responsive Fine-Tuner", theme=gr.themes.Soft()) as self.demo:
            gr.Markdown("# 🎯 Responsive Fine-Tuner")
            gr.Markdown("Fine-tune language models with simple feedback - no coding required!")
            
            with gr.Tabs():
                with gr.TabItem("1. Setup"):
                    with gr.Row():
                        with gr.Column():
                            file_input = gr.File(
                                label="Upload your data file",
                                file_types=[".txt", ".csv"],
                                type="filepath"
                            )
                            
                            task_type = gr.Radio(
                                choices=["Sentiment Analysis", "Topic Classification", "Custom"],
                                label="Task Type",
                                value="Sentiment Analysis"
                            )
                            
                            with gr.Group(visible=False) as custom_labels_group:
                                gr.Markdown("### Define your labels")
                                label1 = gr.Textbox(label="Positive/Class 1", value="Positive")
                                label0 = gr.Textbox(label="Negative/Class 0", value="Negative")
                            
                            initialize_btn = gr.Button("Initialize Project", variant="primary")
                        
                        with gr.Column():
                            setup_status = gr.Markdown("### Status: Waiting for data upload...")
                
                with gr.TabItem("2. Label & Train"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Correct the model's predictions")
                            labeling_df = gr.Dataframe(
                                headers=["Text", "Model Prediction", "Your Label"],
                                datatype=["str", "str", "str"],
                                interactive=True,
                                col_count=(3, "fixed")
                            )
                            
                            with gr.Row():
                                submit_feedback_btn = gr.Button("Submit Feedback & Retrain", variant="primary")
                                refresh_sample_btn = gr.Button("Get New Sample")
                            
                            training_status = gr.Markdown("")
                        
                        with gr.Column():
                            gr.Markdown("### Training Progress")
                            accuracy_plot = gr.LinePlot(
                                x="Cycle",
                                y="Accuracy",
                                title="Model Accuracy Over Feedback Cycles"
                            )
                
                with gr.TabItem("3. Test & Demo"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Try your model live!")
                            demo_input = gr.Textbox(
                                label="Enter text to analyze",
                                placeholder="Type any text here to see the model's prediction..."
                            )
                            demo_btn = gr.Button("Analyze", variant="primary")
                            demo_output = gr.Label(label="Prediction")
                        
                        with gr.Column():
                            gr.Markdown("### Model Information")
                            model_info = gr.Markdown("Model not yet initialized")
            
            # State components
            current_model_state = gr.State()
            feedback_loop_state = gr.State()
            
            # Event handlers will be added in the next phase
            gr.Markdown("---")
            gr.Markdown("### Instructions:")
            gr.Markdown("""
            1. **Upload** your text data (CSV or TXT file)
            2. **Correct** the model's predictions in the labeling tab
            3. **Submit feedback** to fine-tune the model
            4. **Test** your improved model in the demo tab
            """)
    
    def launch(self, share=False):
        """Launch the Gradio interface"""
        self.demo.launch(share=share)

# Create and launch the app
if __name__ == "__main__":
    app = ResponsiveFineTunerApp()
    app.launch()