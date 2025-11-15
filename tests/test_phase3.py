import pytest
import sys
import os
import tempfile
import gradio as gr
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from frontend.app_core import RFTApplication
from frontend.main_app import RFTInterface
from backend.data_processor import ProcessedData
import yaml

class TestPhase3:
    """Test suite for Phase 3 frontend components"""
    
    def setup_method(self):
        """Setup before each test"""
        # Create temporary config
        self.test_config = {
            'model': {
                'base_model': 'distilbert-base-uncased',
                'max_length': 128,
                'num_labels': 2
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 1e-4,
                'max_steps': 10,
                'lora_r': 4,
                'lora_alpha': 8,
                'lora_dropout': 0.05
            },
            'data': {
                'supported_formats': ['.txt', '.csv'],
                'max_file_size_mb': 1,
                'test_split': 0.2
            },
            'ui': {
                'theme': 'soft',
                'batch_size_labeling': 3,
                'update_interval': 500
            }
        }
        
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def test_app_core_initialization(self):
        """Test RFTApplication core initialization"""
        app = RFTApplication(self.config_path)
        
        assert app.config is not None
        assert app.data_processor is not None
        assert app.model_manager is not None
        assert app.current_data is None
        assert app.current_document_index == 0
        
        print("✓ App core initialization test passed")
    
    def test_file_processing_workflow(self):
        """Test file processing workflow"""
        app = RFTApplication(self.config_path)
        
        # Mock file processing
        with patch.object(app.data_processor, 'process_uploaded_files') as mock_process:
            mock_process.return_value = ProcessedData(
                texts=["doc1", "doc2", "doc3"],
                labels=[0, 0, 0],
                metadata=[{}, {}, {}],
                file_sources=["test.txt"]
            )
            
            with patch.object(app.data_processor, 'split_data') as mock_split:
                mock_split.return_value = (
                    ProcessedData(["doc1", "doc2"], [0, 0], [{}, {}], ["test.txt"]),
                    ProcessedData(["doc3"], [0], [{}], ["test.txt"])
                )
                
                result = app.process_uploaded_files(["test.txt"])
                
                assert result["status"] == "success"
                assert result["document_count"] == 3
                assert app.train_data is not None
                assert app.test_data is not None
        
        print("✓ File processing workflow test passed")
    
    def test_document_retrieval(self):
        """Test document retrieval for labeling"""
        app = RFTApplication(self.config_path)
        
        # Setup test data
        app.train_data = ProcessedData(
            texts=["First document", "Second document"],
            labels=[0, 0],
            metadata=[{"id": 1}, {"id": 2}],
            file_sources=["test.txt"]
        )
        
        # Mock prediction
        with patch.object(app.model_manager, 'get_prediction') as mock_pred:
            mock_pred.return_value = {
                "predicted_label": 0,
                "probabilities": [0.8, 0.2],
                "confidence": 0.8
            }
            
            document, doc_data = app.get_next_document()
            
            assert document == "First document"
            assert doc_data["status"] == "success"
            assert doc_data["document_index"] == 0
        
        print("✓ Document retrieval test passed")
    
    def test_feedback_submission(self):
        """Test feedback submission workflow"""
        app = RFTApplication(self.config_path)
        
        # Setup
        app.train_data = ProcessedData(
            texts=["Test document"],
            labels=[0],
            metadata=[{}],
            file_sources=["test.txt"]
        )
        
        # Mock dependencies
        with patch.object(app.model_manager, 'get_prediction') as mock_pred:
            with patch.object(app.model_manager, 'add_feedback') as mock_add_fb:
                with patch.object(app.model_manager, 'incremental_fine_tune') as mock_train:
                    with patch.object(app.model_manager, 'evaluate_model') as mock_eval:
                        
                        mock_pred.return_value = {"predicted_label": 0, "probabilities": [0.7, 0.3]}
                        mock_train.return_value = {"status": "success", "loss": 0.5}
                        mock_eval.return_value = {"accuracy": 0.8}
                        
                        # First, get a document to set up state
                        app.get_next_document()
                        
                        # Submit feedback
                        result = app.submit_feedback("Test document", "Correct", None)
                        
                        assert result["status"] == "success"
                        assert result["feedback_count"] == 1
                        mock_add_fb.assert_called_once()
        
        print("✓ Feedback submission test passed")
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        app = RFTApplication(self.config_path)
        
        # Add some mock history
        app.performance_history = [
            {"feedback_count": 10, "accuracy": 0.7, "timestamp": "2023-01-01"}
        ]
        app.labeling_history = [{"document": "test", "user_label": 0}]
        
        metrics = app.get_performance_metrics()
        
        assert "current_accuracy" in metrics
        assert "labeling_progress" in metrics
        assert "model_info" in metrics
        
        print("✓ Performance metrics test passed")
    
    def test_interface_initialization(self):
        """Test RFTInterface initialization"""
        interface = RFTInterface(self.config_path)
        
        assert interface.app is not None
        assert isinstance(interface.app, RFTApplication)
        
        print("✓ Interface initialization test passed")
    
    def test_interface_setup(self):
        """Test interface setup"""
        interface = RFTInterface(self.config_path)
        
        # Mock the component creation to avoid full Gradio setup
        with patch('frontend.main_app.create_data_upload_interface') as mock_data:
            with patch('frontend.main_app.create_labeling_interface') as mock_label:
                with patch('frontend.main_app.create_performance_interface') as mock_perf:
                    
                    mock_data.return_value = gr.Blocks()
                    mock_label.return_value = gr.Blocks()
                    mock_perf.return_value = gr.Blocks()
                    
                    gradio_interface = interface.setup_interface()
                    
                    assert gradio_interface is not None
                    mock_data.assert_called_once()
                    mock_label.assert_called_once()
                    mock_perf.assert_called_once()
        
        print("✓ Interface setup test passed")
    
    def test_wrapper_methods(self):
        """Test Gradio wrapper methods"""
        interface = RFTInterface(self.config_path)
        
        # Test file processing wrapper
        with patch.object(interface.app, 'process_uploaded_files') as mock_process:
            mock_process.return_value = {
                "status": "success",
                "message": "Test success",
                "document_count": 3,
                "train_count": 2,
                "test_count": 1,
                "file_sources": ["test.txt"]
            }
            
            status, info = interface._process_files_wrapper([], 0.2, 256)
            
            assert status["status"] == "✅ Success"
            assert len(info) > 0
        
        print("✓ Wrapper methods test passed")

def run_phase3_tests():
    """Run all Phase 3 tests"""
    print("🚀 Running Phase 3 Tests...")
    print("=" * 50)
    
    test_suite = TestPhase3()
    
    try:
        test_suite.setup_method()
        test_suite.test_app_core_initialization()
        test_suite.test_file_processing_workflow()
        test_suite.test_document_retrieval()
        test_suite.test_feedback_submission()
        test_suite.test_performance_metrics()
        test_suite.test_interface_initialization()
        test_suite.test_interface_setup()
        test_suite.test_wrapper_methods()
        
        print("=" * 50)
        print("🎉 ALL PHASE 3 TESTS PASSED!")
        print("✓ Application core functionality working")
        print("✓ File processing workflow implemented")
        print("✓ Interactive labeling system ready")
        print("✓ Feedback submission pipeline functional")
        print("✓ Performance monitoring system working")
        print("✓ Gradio interface components created")
        print("✓ Wrapper methods for Gradio integration")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_phase3_tests()