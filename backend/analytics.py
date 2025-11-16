import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from backend.data_processor import ProcessedData

class ModelAnalytics:
    """Comprehensive model analytics and evaluation"""
    
    def __init__(self):
        self.evaluation_history = []
        self.performance_metrics = {}
        
    def comprehensive_evaluation(self, model_manager, test_data) -> Dict[str, Any]:
        """Perform comprehensive model evaluation"""
        try:
            # Get predictions
            predictions = []
            actual_labels = []
            confidences = []
            
            for text, true_label in zip(test_data.texts, test_data.labels):
                pred = model_manager.get_prediction(text)
                predictions.append(pred['predicted_label'])
                actual_labels.append(true_label)
                confidences.append(pred['confidence'])
            
            # Calculate metrics
            accuracy = accuracy_score(actual_labels, predictions)
            precision = precision_score(actual_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(actual_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(actual_labels, predictions, average='weighted', zero_division=0)
            
            # Confidence analysis
            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            # Calibration analysis
            calibration = self._analyze_calibration(predictions, actual_labels, confidences)
            
            # Confusion matrix
            cm = confusion_matrix(actual_labels, predictions)
            
            evaluation_result = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'average_confidence': avg_confidence,
                'confidence_std': confidence_std,
                'calibration_metrics': calibration,
                'confusion_matrix': cm.tolist(),
                'sample_count': len(predictions)
            }
            
            self.evaluation_history.append(evaluation_result)
            self._update_performance_trends()
            
            logger.info(f"Comprehensive evaluation completed: Accuracy = {accuracy:.4f}")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_calibration(self, predictions: List[int], actuals: List[int], 
                           confidences: List[float]) -> Dict[str, float]:
        """Analyze model calibration"""
        try:
            # Group predictions by confidence bins
            bins = np.linspace(0, 1, 11)
            bin_accuracies = []
            bin_confidences = []
            
            for i in range(len(bins) - 1):
                low, high = bins[i], bins[i+1]
                mask = (np.array(confidences) >= low) & (np.array(confidences) < high)
                
                if np.sum(mask) > 0:
                    bin_acc = accuracy_score(np.array(actuals)[mask], np.array(predictions)[mask])
                    bin_conf = np.mean(np.array(confidences)[mask])
                    bin_accuracies.append(bin_acc)
                    bin_confidences.append(bin_conf)
            
            # Calculate calibration error
            if bin_accuracies and bin_confidences:
                calibration_error = np.mean(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)))
            else:
                calibration_error = 0.0
            
            return {
                'expected_calibration_error': calibration_error,
                'reliability_diagram': {
                    'confidence_bins': bins.tolist(),
                    'accuracies': bin_accuracies,
                    'confidences': bin_confidences
                }
            }
            
        except Exception as e:
            logger.error(f"Calibration analysis failed: {e}")
            return {'expected_calibration_error': 0.0}
    
    def _update_performance_trends(self):
        """Update performance trend analysis"""
        if len(self.evaluation_history) < 2:
            return
        
        recent_evals = self.evaluation_history[-5:]  # Last 5 evaluations
        
        trends = {}
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics:
            values = [e.get(metric, 0) for e in recent_evals]
            if len(values) >= 2:
                trend = (values[-1] - values[0]) / len(values)
                trends[f'{metric}_trend'] = trend
        
        self.performance_metrics['trends'] = trends
    
    def create_analytics_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive analytics dashboard"""
        if not self.evaluation_history:
            return {'status': 'no_data'}
        
        latest_eval = self.evaluation_history[-1]
        
        dashboard = {
            'current_performance': {
                'accuracy': latest_eval.get('accuracy', 0),
                'precision': latest_eval.get('precision', 0),
                'recall': latest_eval.get('recall', 0),
                'f1_score': latest_eval.get('f1_score', 0),
                'average_confidence': latest_eval.get('average_confidence', 0)
            },
            'trends': self.performance_metrics.get('trends', {}),
            'calibration': latest_eval.get('calibration_metrics', {}),
            'evaluation_count': len(self.evaluation_history),
            'last_evaluation': latest_eval.get('timestamp')
        }
        
        return dashboard
    
    def create_performance_plots(self) -> Dict[str, go.Figure]:
        """Create Plotly figures for performance visualization"""
        if len(self.evaluation_history) < 2:
            return {}
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.evaluation_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        figures = {}
        
        # Performance metrics over time
        fig_metrics = make_subplots(rows=2, cols=2, 
                                  subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'))
        
        metrics = [('accuracy', 1, 1), ('precision', 1, 2), 
                  ('recall', 2, 1), ('f1_score', 2, 2)]
        
        for metric, row, col in metrics:
            fig_metrics.add_trace(
                go.Scatter(x=df['timestamp'], y=df[metric], 
                          name=metric.replace('_', ' ').title(),
                          line=dict(width=3)),
                row=row, col=col
            )
        
        fig_metrics.update_layout(height=600, title_text="Performance Metrics Over Time")
        figures['performance_metrics'] = fig_metrics
        
        # Confidence distribution
        latest_confidences = [pred.get('confidence', 0) for pred in 
                             self.evaluation_history[-1].get('predictions', [])]
        
        fig_confidence = go.Figure()
        fig_confidence.add_trace(go.Histogram(x=latest_confidences, nbinsx=20,
                                             name='Confidence Distribution'))
        fig_confidence.update_layout(title="Prediction Confidence Distribution",
                                   xaxis_title="Confidence",
                                   yaxis_title="Count")
        figures['confidence_distribution'] = fig_confidence
        
        return figures

class DataQualityAnalyzer:
    """Analyze data quality and provide insights"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def analyze_data_quality(self, processed_data: ProcessedData) -> Dict[str, Any]:
        """Analyze quality of the processed data"""
        try:
            texts = processed_data.texts
            
            # Text length analysis
            text_lengths = [len(text.split()) for text in texts]
            avg_length = np.mean(text_lengths)
            length_std = np.std(text_lengths)
            
            # Vocabulary analysis
            all_words = ' '.join(texts).split()
            vocab_size = len(set(all_words))
            avg_word_length = np.mean([len(word) for word in all_words])
            
            # Label distribution (if available)
            label_dist = {}
            if processed_data.labels and any(l != 0 for l in processed_data.labels):
                unique_labels = set(processed_data.labels)
                for label in unique_labels:
                    label_dist[f"class_{label}"] = processed_data.labels.count(label)
            
            quality_metrics = {
                'sample_count': len(texts),
                'average_text_length': avg_length,
                'text_length_std': length_std,
                'vocabulary_size': vocab_size,
                'average_word_length': avg_word_length,
                'label_distribution': label_dist,
                'data_quality_score': self._calculate_quality_score(avg_length, vocab_size, len(texts))
            }
            
            self.quality_metrics = quality_metrics
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_quality_score(self, avg_length: float, vocab_size: int, 
                               sample_count: int) -> float:
        """Calculate overall data quality score"""
        # Normalize metrics
        length_score = min(avg_length / 50, 1.0)  # Ideal length around 50 words
        vocab_score = min(vocab_size / 1000, 1.0)  # Reasonable vocabulary size
        sample_score = min(sample_count / 100, 1.0)  # Reasonable sample count
        
        # Weighted average
        quality_score = (length_score * 0.4 + vocab_score * 0.3 + sample_score * 0.3)
        return round(quality_score, 3)
    
    def get_data_insights(self) -> List[str]:
        """Get actionable insights from data quality analysis"""
        insights = []
        metrics = self.quality_metrics
        
        if not metrics or 'error' in metrics:
            return ["No data quality analysis available"]
        
        # Sample count insights
        if metrics['sample_count'] < 50:
            insights.append("⚠️ Low sample count - consider adding more data")
        elif metrics['sample_count'] > 1000:
            insights.append("✅ Good sample count for training")
        
        # Text length insights
        if metrics['average_text_length'] < 10:
            insights.append("⚠️ Texts are very short - may lack context")
        elif metrics['average_text_length'] > 200:
            insights.append("⚠️ Texts are very long - consider splitting")
        else:
            insights.append("✅ Text lengths are appropriate")
        
        # Vocabulary insights
        if metrics['vocabulary_size'] < 100:
            insights.append("⚠️ Limited vocabulary diversity")
        else:
            insights.append("✅ Good vocabulary diversity")
        
        # Label distribution insights
        label_dist = metrics.get('label_distribution', {})
        if len(label_dist) > 1:
            counts = list(label_dist.values())
            balance_ratio = min(counts) / max(counts)
            if balance_ratio < 0.3:
                insights.append("⚠️ Class imbalance detected - consider balancing")
            else:
                insights.append("✅ Classes are reasonably balanced")
        
        return insights
