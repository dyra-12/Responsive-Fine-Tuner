import gradio as gr
import plotly.graph_objects as go
from typing import Dict, List, Any
import pandas as pd

def create_analytics_interface(app, analytics_callback: Callable) -> gr.Blocks:
    """Create advanced analytics interface"""
    
    with gr.Blocks() as interface:
        gr.Markdown("## 📈 Advanced Analytics")
        gr.Markdown("Comprehensive model performance and data quality analytics.")
        
        with gr.Tabs():
            with gr.TabItem("Model Performance"):
                with gr.Row():
                    with gr.Column(scale=1):
                        performance_metrics = gr.JSON(
                            label="Performance Metrics",
                            container=True
                        )
                        
                        run_analysis_btn = gr.Button(
                            "Run Comprehensive Analysis",
                            variant="primary"
                        )
                    
                    with gr.Column(scale=2):
                        performance_plot = gr.Plot(
                            label="Performance Trends"
                        )
                        
                        confidence_plot = gr.Plot(
                            label="Confidence Analysis"
                        )
            
            with gr.TabItem("Data Quality"):
                with gr.Row():
                    with gr.Column(scale=1):
                        data_quality_metrics = gr.JSON(
                            label="Data Quality Metrics",
                            container=True
                        )
                        
                        data_insights = gr.JSON(
                            label="Data Insights",
                            container=True
                        )
                    
                    with gr.Column(scale=2):
                        text_length_plot = gr.Plot(
                            label="Text Length Distribution"
                        )
                        
                        label_distribution_plot = gr.Plot(
                            label="Label Distribution"
                        )
            
            with gr.TabItem("Training Analytics"):
                with gr.Row():
                    with gr.Column(scale=1):
                        training_analytics = gr.JSON(
                            label="Training Analytics",
                            container=True
                        )
                    
                    with gr.Column(scale=2):
                        learning_rate_plot = gr.Plot(
                            label="Learning Rate Adaptation"
                        )
                        
                        loss_trend_plot = gr.Plot(
                            label="Training Loss Trend"
                        )
        
        # Event handlers
        run_analysis_btn.click(
            fn=analytics_callback,
            outputs=[
                performance_metrics, performance_plot, confidence_plot,
                data_quality_metrics, data_insights, text_length_plot, label_distribution_plot,
                training_analytics, learning_rate_plot, loss_trend_plot
            ]
        )
    
    return interface

def create_analytics_display(analytics_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create formatted analytics display"""
    if 'error' in analytics_data:
        return {"error": analytics_data['error']}
    
    return {
        "current_performance": analytics_data.get('current_performance', {}),
        "data_quality": analytics_data.get('data_quality', {}),
        "training_analytics": analytics_data.get('training_analytics', {}),
        "timestamp": analytics_data.get('timestamp', 'N/A')
    }