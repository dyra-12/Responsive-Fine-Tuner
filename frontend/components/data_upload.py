import gradio as gr
import os
from typing import Dict, Any, Callable

def create_data_upload_interface(app, process_callback: Callable) -> gr.Blocks:
    """Create the data upload interface component"""
    
    with gr.Blocks() as interface:
        gr.Markdown("## 📁 Data Upload & Processing")
        gr.Markdown("Upload your text documents or CSV files for model fine-tuning.")
        
        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    file_count="multiple",
                    file_types=[".txt", ".csv"],
                    label="Upload Documents",
                    elem_id="file_upload"
                )
                
                process_btn = gr.Button(
                    "Process Uploaded Files", 
                    variant="primary",
                    size="lg"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    test_split_slider = gr.Slider(
                        minimum=0.1,
                        maximum=0.5,
                        value=0.2,
                        step=0.05,
                        label="Test Split Ratio"
                    )
                    
                    max_length_slider = gr.Slider(
                        minimum=128,
                        maximum=512,
                        value=256,
                        step=64,
                        label="Maximum Text Length"
                    )
            
            with gr.Column(scale=1):
                upload_status = gr.JSON(
                    label="Upload Status",
                    show_label=True
                )
                
                # Gradio is deprecating `col_count` in favor of `column_count`.
                # Keep compatibility with both.
                try:
                    dataset_info = gr.Dataframe(
                        headers=["Metric", "Value"],
                        label="Dataset Information",
                        row_count=5,
                        column_count=2,
                    )
                except TypeError:
                    dataset_info = gr.Dataframe(
                        headers=["Metric", "Value"],
                        label="Dataset Information",
                        row_count=5,
                        col_count=2,
                    )
        
        # Event handlers
        process_btn.click(
            fn=process_callback,
            inputs=[file_upload, test_split_slider, max_length_slider],
            outputs=[upload_status, dataset_info]
        )
    
    return interface

def create_upload_status_display(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create formatted upload status display"""
    if result["status"] == "success":
        return {
            "status": "✅ Success",
            "message": result["message"],
            "documents_processed": result["document_count"],
            "training_samples": result["train_count"],
            "test_samples": result["test_count"]
        }
    else:
        return {
            "status": "❌ Error",
            "message": result["message"]
        }

def create_dataset_info_df(result: Dict[str, Any]) -> list:
    """Create dataset information dataframe"""
    if result["status"] != "success":
        return [["Status", "No data processed"]]
    
    info = [
        ["Total Documents", result["document_count"]],
        ["Training Samples", result["train_count"]],
        ["Test Samples", result["test_count"]],
        ["File Sources", len(result["file_sources"])],
        ["Status", "Ready for Labeling"]
    ]
    
    return info