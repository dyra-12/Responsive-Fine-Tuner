import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np

def create_labeling_interface(app, next_doc_callback: Callable, feedback_callback: Callable) -> gr.Blocks:
    """Create the interactive labeling interface"""
    
    with gr.Blocks() as interface:
        gr.Markdown("## 🎯 Interactive Labeling")
        gr.Markdown("Review model predictions and provide feedback to improve the model.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Document display
                # Some Gradio versions don't support `show_copy_button` on Textbox.
                try:
                    document_display = gr.Textbox(
                        label="Current Document",
                        lines=8,
                        max_lines=12,
                        show_copy_button=True,
                        container=True,
                    )
                except TypeError:
                    document_display = gr.Textbox(
                        label="Current Document",
                        lines=8,
                        max_lines=12,
                        container=True,
                    )
                
                # Progress info
                with gr.Row():
                    progress_text = gr.Textbox(
                        label="Progress",
                        interactive=False,
                        scale=3
                    )
                    progress_bar = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=0,
                        label="Completion %",
                        interactive=False,
                        show_label=True,
                        scale=7
                    )
            
            with gr.Column(scale=1):
                # Model prediction
                model_prediction = gr.Label(
                    label="Model Prediction",
                    num_top_classes=2,
                    container=True
                )
                
                # Confidence visualization
                confidence_plot = gr.Plot(
                    label="Prediction Confidence",
                    container=True
                )
                
                # User feedback section
                with gr.Group():
                    user_feedback = gr.Radio(
                        choices=["Correct", "Incorrect"],
                        label="Is this prediction correct?",
                        container=True
                    )
                    
                    correct_label = gr.Dropdown(
                        choices=app.available_labels,
                        label="Correct Label (if prediction is wrong)",
                        visible=False,
                        container=True
                    )
                
                submit_btn = gr.Button(
                    "Submit Feedback & Next", 
                    variant="primary",
                    size="lg"
                )
                
                next_btn = gr.Button(
                    "Skip & Next Document",
                    variant="secondary"
                )
        
        # Hidden components for state management
        current_doc_data = gr.State()
        
        # Event handlers
        user_feedback.change(
            fn=lambda x: gr.update(visible=(x == "Incorrect")),
            inputs=user_feedback,
            outputs=correct_label
        )
        
        submit_btn.click(
            fn=feedback_callback,
            inputs=[document_display, user_feedback, correct_label],
            outputs=[progress_text, progress_bar, current_doc_data]
        ).then(
            fn=next_doc_callback,
            outputs=[document_display, model_prediction, confidence_plot, progress_text, progress_bar, current_doc_data]
        )
        
        next_btn.click(
            fn=next_doc_callback,
            outputs=[document_display, model_prediction, confidence_plot, progress_text, progress_bar, current_doc_data]
        )
    
    return interface

def create_prediction_display(prediction_data: Dict[str, Any]) -> Tuple[Dict, go.Figure]:
    """Create model prediction display and confidence plot"""
    if "error" in prediction_data:
        # Error case
        label_display = {
            "Error": 1.0
        }
        
        # Create error plot
        fig = go.Figure()
        fig.add_annotation(
            text="Prediction Error",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Prediction Confidence",
            xaxis_title="Classes",
            yaxis_title="Confidence",
            showlegend=False
        )
        
    else:
        # Normal prediction case
        labels = [f"Class {i}" for i in range(len(prediction_data["probabilities"]))]
        confidences = prediction_data["probabilities"]
        
        label_display = {
            labels[i]: float(confidences[i]) 
            for i in range(len(labels))
        }
        
        # Create confidence plot
        fig = px.bar(
            x=labels,
            y=confidences,
            title="Prediction Confidence",
            labels={'x': 'Classes', 'y': 'Confidence'}
        )
        fig.update_traces(
            marker_color=['blue' if i == prediction_data["predicted_label"] else 'lightblue' 
                         for i in range(len(labels))]
        )
        fig.update_layout(
            showlegend=False,
            yaxis=dict(range=[0, 1])
        )
    
    return label_display, fig

def create_progress_display(current_index: int, total_documents: int) -> Tuple[str, float]:
    """Create progress text and percentage"""
    if total_documents == 0:
        return "No documents available", 0
    
    percentage = (current_index / total_documents) * 100
    progress_text = f"Document {current_index + 1} of {total_documents} ({percentage:.1f}%)"
    
    return progress_text, percentage