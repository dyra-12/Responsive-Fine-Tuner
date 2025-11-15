import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Callable
import pandas as pd
from datetime import datetime

def create_performance_interface(app, metrics_callback: Callable) -> gr.Blocks:
    """Create the performance monitoring interface"""
    
    with gr.Blocks() as interface:
        gr.Markdown("## 📊 Performance Monitoring")
        gr.Markdown("Track model performance and training progress in real-time.")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Current metrics
                current_metrics = gr.JSON(
                    label="Current Metrics",
                    container=True
                )
                
                # Model information
                model_info = gr.JSON(
                    label="Model Information",
                    container=True
                )
            
            with gr.Column(scale=2):
                # Performance plots
                with gr.Tabs():
                    with gr.TabItem("Accuracy Over Time"):
                        accuracy_plot = gr.Plot(
                            label="Accuracy Progress"
                        )
                    
                    with gr.TabItem("Training History"):
                        training_plot = gr.Plot(
                            label="Training Progress"
                        )
                    
                    with gr.TabItem("Label Distribution"):
                        label_plot = gr.Plot(
                            label="Label Distribution"
                        )
        
        # Control buttons
        with gr.Row():
            refresh_btn = gr.Button(
                "Refresh Metrics", 
                variant="secondary"
            )
            export_btn = gr.Button(
                "Export Training Data",
                variant="primary"
            )
        
        # Event handlers
        refresh_btn.click(
            fn=metrics_callback,
            outputs=[current_metrics, model_info, accuracy_plot, training_plot, label_plot]
        )
    
    return interface

def create_performance_plots(metrics: Dict[str, Any]) -> List[go.Figure]:
    """Create performance monitoring plots"""
    
    # Accuracy over time plot
    accuracy_fig = go.Figure()
    if metrics.get("performance_history"):
        history = metrics["performance_history"]
        feedback_counts = [entry["feedback_count"] for entry in history]
        accuracies = [entry["accuracy"] for entry in history]
        
        accuracy_fig.add_trace(go.Scatter(
            x=feedback_counts,
            y=accuracies,
            mode='lines+markers',
            name='Test Accuracy',
            line=dict(color='blue', width=3)
        ))
        
        accuracy_fig.update_layout(
            title="Model Accuracy vs Feedback Samples",
            xaxis_title="Feedback Samples",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1])
        )
    else:
        accuracy_fig.add_annotation(
            text="No training history yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        accuracy_fig.update_layout(title="Model Accuracy Over Time")
    
    # Training history plot
    training_fig = go.Figure()
    if metrics.get("model_info", {}).get("training_sessions", 0) > 0:
        training_sessions = metrics["model_info"]["training_sessions"]
        training_fig.add_trace(go.Indicator(
            mode="number",
            value=training_sessions,
            title={"text": "Training Sessions"},
            number={'suffix': " sessions"}
        ))
    else:
        training_fig.add_annotation(
            text="No training sessions yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        training_fig.update_layout(title="Training History")
    
    # Label distribution plot
    label_fig = go.Figure()
    if metrics.get("labeled_count", 0) > 0:
        # This would need actual label distribution data
        # For now, show a simple progress indicator
        labeled = metrics["labeled_count"]
        total = metrics["total_count"]
        
        label_fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=labeled,
            title={'text': f"Labeled Samples ({labeled}/{total})"},
            gauge={
                'axis': {'range': [0, total]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, total], 'color': "lightgray"}
                ]
            }
        ))
    else:
        label_fig.add_annotation(
            text="No labeling data yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        label_fig.update_layout(title="Label Distribution")
    
    return [accuracy_fig, training_fig, label_fig]

def create_current_metrics_display(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Create formatted current metrics display"""
    if "error" in metrics:
        return {"error": metrics["error"]}
    
    return {
        "current_accuracy": f"{metrics.get('current_accuracy', 0):.4f}",
        "labeling_progress": f"{metrics.get('labeling_progress', 0):.1f}%",
        "labeled_samples": metrics.get("labeled_count", 0),
        "total_samples": metrics.get("total_count", 0),
        "feedback_samples": metrics.get("feedback_count", 0),
        "training_sessions": metrics.get("training_sessions", 0)
    }