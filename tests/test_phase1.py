import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.config import ConfigManager
from backend.model_manager import ModelManager
import tempfile
import yaml

class TestPhase1:
    """Test suite for Phase 1 components"""
    
    def setup_method(self):
        """Setup before each test"""
        # Create temporary config for testing
        self.test_config = {
            'model': {
                'base_model': 'distilbert-base-uncased',
                'max_length': 128,  # Smaller for testing
                'num_labels': 2
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 1e-4,
                'max_steps': 10,
                'lora_r': 4,  # Smaller for testing
                'lora_alpha': 8,
                'lora_dropout': 0.05
            },
            'data': {
                'supported_formats': ['.txt', '.csv'],
                'max_file_size_mb': 1,
                'test_split': 0.1
            },
            'ui': {
                'theme': 'soft',
                'batch_size_labeling': 3,
                'update_interval': 500
            }
        }
        
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def test_config_loading(self):
        """Test configuration manager loads correctly"""
        config_manager = ConfigManager(self.config_path)
        config = config_manager.get_config()
        
        assert config['model'].base_model == 'distilbert-base-uncased'
        assert config['training'].batch_size == 2
        assert config['data'].max_file_size_mb == 1
        assert config['ui'].theme == 'soft'
        
        print("✓ Config loading test passed")
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        config_manager = ConfigManager(self.config_path)
        model_manager = ModelManager(config_manager)
        
        # Test model info
        model_info = model_manager.get_model_info()
        assert model_info['status'] == 'initialized'
        assert model_info['base_model'] == 'distilbert-base-uncased'
        assert model_info['lora_enabled'] == True
        assert model_info['trainable_percentage'] > 0
        assert model_info['trainable_percentage'] < 100
        
        print("✓ Model initialization test passed")
    
    def test_prediction_functionality(self):
        """Test that predictions work"""
        config_manager = ConfigManager(self.config_path)
        model_manager = ModelManager(config_manager)
        
        # Test single prediction
        test_text = "This is a test document for prediction."
        prediction = model_manager.get_prediction(test_text)
        
        assert 'predicted_label' in prediction
        assert 'probabilities' in prediction
        assert 'confidence' in prediction
        assert prediction['text'] == test_text
        assert isinstance(prediction['predicted_label'], int)
        assert len(prediction['probabilities']) == 2  # num_labels
        
        print("✓ Prediction functionality test passed")
    
    def test_batch_prediction(self):
        """Test batch predictions"""
        config_manager = ConfigManager(self.config_path)
        model_manager = ModelManager(config_manager)
        
        test_texts = [
            "First test document.",
            "Second test document with more text.",
            "Third test document for batch processing."
        ]
        
        predictions = model_manager.batch_predict(test_texts)
        
        assert len(predictions) == len(test_texts)
        for i, prediction in enumerate(predictions):
            assert prediction['text'] == test_texts[i]
            assert 'confidence' in prediction
        
        print("✓ Batch prediction test passed")
    
    def test_feedback_storage(self):
        """Test feedback storage functionality"""
        config_manager = ConfigManager(self.config_path)
        model_manager = ModelManager(config_manager)
        
        initial_feedback_count = len(model_manager.feedback_data)
        
        # Add feedback
        test_text = "Feedback test document"
        test_prediction = model_manager.get_prediction(test_text)
        model_manager.add_feedback(test_text, 1, test_prediction)
        
        assert len(model_manager.feedback_data) == initial_feedback_count + 1
        assert model_manager.feedback_data[-1]['text'] == test_text
        assert model_manager.feedback_data[-1]['user_label'] == 1
        
        print("✓ Feedback storage test passed")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        config_manager = ConfigManager(self.config_path)
        model_manager = ModelManager(config_manager)
        
        # Test with empty string
        prediction = model_manager.get_prediction("")
        assert 'error' in prediction or prediction['confidence'] == 0.0
        
        print("✓ Error handling test passed")

def run_phase1_tests():
    """Run all Phase 1 tests"""
    print("🚀 Running Phase 1 Tests...")
    print("=" * 50)
    
    test_suite = TestPhase1()
    
    try:
        test_suite.setup_method()
        test_suite.test_config_loading()
        test_suite.test_model_initialization()
        test_suite.test_prediction_functionality()
        test_suite.test_batch_prediction()
        test_suite.test_feedback_storage()
        test_suite.test_error_handling()
        
        print("=" * 50)
        print("🎉 ALL PHASE 1 TESTS PASSED!")
        print("✓ Configuration system working")
        print("✓ Model loading and initialization working")
        print("✓ Prediction system functional")
        print("✓ Feedback storage implemented")
        print("✓ Error handling in place")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    run_phase1_tests()