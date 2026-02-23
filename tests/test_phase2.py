import pytest
import sys
import os
import tempfile
import pandas as pd
from unittest.mock import Mock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.config import ConfigManager
from backend.data_processor import DataProcessor, ProcessedData
from backend.enhanced_model_manager import EnhancedModelManager, FeedbackDataset
import yaml

class TestPhase2:
    """Test suite for Phase 2 components"""
    
    def setup_method(self):
        """Setup before each test"""
        # Create temporary config for testing
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
        
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
        
        self.config_manager = ConfigManager(self.config_path)
        
        # Create test data files
        self.create_test_data_files()
    
    def create_test_data_files(self):
        """Create test data files for processing"""
        # Create test text file
        self.text_file_path = os.path.join(self.temp_dir, "test_documents.txt")
        with open(self.text_file_path, 'w', encoding='utf-8') as f:
            f.write("This is the first test document.\n\n")
            f.write("This is the second test document with more content for testing the data processing pipeline.\n\n")
            f.write("Third document for testing text file processing capabilities.\n\n")
        
        # Create test CSV file
        self.csv_file_path = os.path.join(self.temp_dir, "test_data.csv")
        csv_data = {
            'text': [
                "First CSV document for testing",
                "Second CSV document with different content",
                "Third CSV document for data processing"
            ],
            'label': [0, 1, 0]
        }
        df = pd.DataFrame(csv_data)
        df.to_csv(self.csv_file_path, index=False)
    
    def test_data_processor_initialization(self):
        """Test data processor initialization"""
        data_processor = DataProcessor(self.config_manager)
        
        assert data_processor.config == self.config_manager
        assert data_processor.supported_formats == ['.txt', '.csv']
        assert data_processor.max_file_size == 1024 * 1024  # 1MB in bytes
        
        print("✓ Data processor initialization test passed")
    
    def test_file_validation(self):
        """Test file validation functionality"""
        data_processor = DataProcessor(self.config_manager)
        
        # Test valid file
        assert data_processor.validate_file(self.text_file_path) == True
        
        # Test invalid extension
        invalid_file = os.path.join(self.temp_dir, "test.pdf")
        open(invalid_file, 'w').close()
        assert data_processor.validate_file(invalid_file) == False
        
        print("✓ File validation test passed")
    
    def test_text_file_processing(self):
        """Test processing of text files"""
        data_processor = DataProcessor(self.config_manager)
        
        documents = data_processor.process_text_file(self.text_file_path)
        
        assert len(documents) == 3
        assert all('text' in doc for doc in documents)
        assert all('source_file' in doc for doc in documents)
        assert all('document_id' in doc for doc in documents)
        
        # Check content
        texts = [doc['text'] for doc in documents]
        assert "first test document" in texts[0]
        assert "second test document" in texts[1]
        
        print("✓ Text file processing test passed")
    
    def test_csv_file_processing(self):
        """Test processing of CSV files"""
        data_processor = DataProcessor(self.config_manager)
        
        documents = data_processor.process_csv_file(self.csv_file_path)
        
        assert len(documents) == 3
        assert all('text' in doc for doc in documents)
        assert all('source_file' in doc for doc in documents)
        assert all('row_data' in doc for doc in documents)
        # If a label column exists, it should be preserved in metadata.
        assert all('label' in doc for doc in documents)
        
        print("✓ CSV file processing test passed")

    def test_csv_labels_opt_in(self):
        """Test that labels are only populated when use_labels=True"""
        data_processor = DataProcessor(self.config_manager)

        # Default: treat uploaded data as unlabeled for interactive labeling
        processed_default = data_processor.process_uploaded_files([self.csv_file_path], use_labels=False)
        assert processed_default.labels == [0, 0, 0]

        # Opt-in: use labels from CSV label column
        processed_labeled = data_processor.process_uploaded_files([self.csv_file_path], use_labels=True)
        assert processed_labeled.labels == [0, 1, 0]

        print("✓ CSV label opt-in test passed")
    
    def test_data_splitting(self):
        """Test train-test split functionality"""
        data_processor = DataProcessor(self.config_manager)
        
        # Create sample processed data
        processed_data = ProcessedData(
            texts=["doc1", "doc2", "doc3", "doc4", "doc5", "doc6", "doc7", "doc8", "doc9", "doc10"],
            labels=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            metadata=[{} for _ in range(10)],
            file_sources=["test.txt"]
        )
        
        train_data, test_data = data_processor.split_data(processed_data, test_size=0.3)
        
        assert len(train_data.texts) == 7  # 70% of 10
        assert len(test_data.texts) == 3   # 30% of 10
        assert len(train_data.texts) + len(test_data.texts) == len(processed_data.texts)
        
        print("✓ Data splitting test passed")
    
    def test_enhanced_model_manager(self):
        """Test enhanced model manager functionality"""
        model_manager = EnhancedModelManager(self.config_manager)
        
        # Test model info
        model_info = model_manager.get_model_info()
        assert model_info['status'] == 'initialized'
        assert model_info['training_sessions'] == 0
        
        print("✓ Enhanced model manager test passed")
    
    def test_dataset_preparation(self):
        """Test dataset preparation"""
        model_manager = EnhancedModelManager(self.config_manager)
        
        # Create sample processed data
        processed_data = ProcessedData(
            texts=["This is a test document.", "Another test document for training."],
            labels=[0, 1],
            metadata=[{}, {}],
            file_sources=["test.txt"]
        )
        
        dataset = model_manager.prepare_dataset(processed_data)
        
        assert len(dataset) == 2
        assert hasattr(dataset, 'texts')
        assert hasattr(dataset, 'labels')
        
        # Test dataset item
        item = dataset[0]
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'labels' in item
        
        print("✓ Dataset preparation test passed")
    
    def test_feedback_storage_and_training(self):
        """Test feedback storage and incremental training"""
        model_manager = EnhancedModelManager(self.config_manager)
        
        # Add feedback
        initial_count = len(model_manager.feedback_data)
        test_text = "Feedback test document"
        test_prediction = model_manager.get_prediction(test_text)
        
        model_manager.add_feedback(test_text, 1, test_prediction)
        
        assert len(model_manager.feedback_data) == initial_count + 1
        assert model_manager.feedback_data[-1]['user_label'] == 1
        
        # Test incremental training with mock
        with patch.object(model_manager, 'fine_tune_step') as mock_fine_tune:
            mock_fine_tune.return_value = {"status": "success", "loss": 0.5}
            
            result = model_manager.incremental_fine_tune(model_manager.feedback_data)
            
            assert result["status"] == "success"
            mock_fine_tune.assert_called_once()
        
        print("✓ Feedback storage and training test passed")
    
    def test_model_evaluation(self):
        """Test model evaluation functionality"""
        model_manager = EnhancedModelManager(self.config_manager)
        
        # Create test data
        test_data = ProcessedData(
            texts=["Evaluation document one", "Evaluation document two"],
            labels=[0, 1],
            metadata=[{}, {}],
            file_sources=["test.txt"]
        )
        
        evaluation_result = model_manager.evaluate_model(test_data)
        
        assert 'accuracy' in evaluation_result
        assert 'total_samples' in evaluation_result
        assert evaluation_result['total_samples'] == 2
        
        print("✓ Model evaluation test passed")
    
    def test_data_save_load(self):
        """Test data saving and loading"""
        data_processor = DataProcessor(self.config_manager)
        
        # Create sample data
        original_data = ProcessedData(
            texts=["Save test doc 1", "Save test doc 2"],
            labels=[0, 1],
            metadata=[{"id": 1}, {"id": 2}],
            file_sources=["test.txt"]
        )
        
        # Save data
        save_path = os.path.join(self.temp_dir, "test_save.json")
        data_processor.save_processed_data(original_data, save_path)
        
        # Load data
        loaded_data = data_processor.load_processed_data(save_path)
        
        assert loaded_data is not None
        assert len(loaded_data.texts) == len(original_data.texts)
        assert loaded_data.texts == original_data.texts
        assert loaded_data.labels == original_data.labels
        
        print("✓ Data save/load test passed")

def run_phase2_tests():
    """Run all Phase 2 tests"""
    print("🚀 Running Phase 2 Tests...")
    print("=" * 50)
    
    test_suite = TestPhase2()
    
    try:
        test_suite.setup_method()
        test_suite.test_data_processor_initialization()
        test_suite.test_file_validation()
        test_suite.test_text_file_processing()
        test_suite.test_csv_file_processing()
        test_suite.test_data_splitting()
        test_suite.test_enhanced_model_manager()
        test_suite.test_dataset_preparation()
        test_suite.test_feedback_storage_and_training()
        test_suite.test_model_evaluation()
        test_suite.test_data_save_load()
        
        print("=" * 50)
        print("🎉 ALL PHASE 2 TESTS PASSED!")
        print("✓ Data processing pipeline working")
        print("✓ File validation and processing functional")
        print("✓ Enhanced model manager with training capabilities")
        print("✓ Dataset preparation and splitting implemented")
        print("✓ Model evaluation system ready")
        print("✓ Data persistence (save/load) working")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_phase2_tests()