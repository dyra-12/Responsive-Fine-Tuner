import pytest
import sys
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.advanced_trainer import AdaptiveLearningRate, SmartSampling, AdvancedModelManager
from backend.analytics import ModelAnalytics, DataQualityAnalyzer
from backend.optimizations import MemoryOptimizer, CachingSystem, BackgroundTrainer
from backend.data_processor import ProcessedData
import yaml

class TestPhase4:
    """Test suite for Phase 4 advanced features"""
    
    def setup_method(self):
        """Setup before each test"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = ProcessedData(
            texts=["Test document one", "Test document two", "Test document three"],
            labels=[0, 1, 0],
            metadata=[{}, {}, {}],
            file_sources=["test.txt"]
        )
    
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate scheduler"""
        lr_scheduler = AdaptiveLearningRate(initial_lr=1e-4)
        
        # Test initial state
        assert lr_scheduler.current_lr == 1e-4
        
        # Test improvement (should increase LR)
        lr_scheduler.update(accuracy=0.8, loss=0.3)
        lr_scheduler.update(accuracy=0.85, loss=0.25)
        new_lr = lr_scheduler.update(accuracy=0.9, loss=0.2)
        
        assert new_lr > 1e-4  # Should increase
        
        # Test worsening (should decrease LR)
        lr_scheduler.update(accuracy=0.7, loss=0.5)
        lr_scheduler.update(accuracy=0.65, loss=0.6)
        final_lr = lr_scheduler.update(accuracy=0.6, loss=0.7)
        
        assert final_lr < new_lr  # Should decrease
        
        print("✓ Adaptive learning rate test passed")
    
    def test_smart_sampling(self):
        """Test smart sampling strategies"""
        sampler = SmartSampling()
        
        # Test uncertainty calculation
        probs_uncertain = [0.5, 0.5]
        probs_confident = [0.9, 0.1]
        
        uncertainty_high = sampler.calculate_uncertainty(probs_uncertain)
        uncertainty_low = sampler.calculate_uncertainty(probs_confident)
        
        assert uncertainty_high > uncertainty_low
        
        # Test sample selection
        texts = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        sampler.update_sample_uncertainty("doc1", [0.1, 0.9])  # Low uncertainty
        sampler.update_sample_uncertainty("doc2", [0.5, 0.5])  # High uncertainty
        sampler.update_sample_uncertainty("doc3", [0.4, 0.6])  # Medium uncertainty
        
        selected = sampler.get_most_uncertain_samples(texts, n=2)
        
        assert "doc2" in selected  # Most uncertain
        assert len(selected) == 2
        
        print("✓ Smart sampling test passed")
    
    def test_model_analytics(self):
        """Test model analytics and evaluation"""
        analytics = ModelAnalytics()
        
        # Mock predictions and actuals
        predictions = [0, 1, 0, 1, 0]
        actuals = [0, 1, 1, 1, 0]
        confidences = [0.9, 0.8, 0.6, 0.7, 0.85]
        
        # Mock model manager
        mock_manager = Mock()
        mock_manager.get_prediction.side_effect = lambda text: {
            'predicted_label': predictions[len(analytics.evaluation_history) % len(predictions)],
            'confidence': confidences[len(analytics.evaluation_history) % len(confidences)]
        }
        
        # Mock test data
        mock_test_data = Mock()
        mock_test_data.texts = ["text1", "text2", "text3", "text4", "text5"]
        mock_test_data.labels = actuals
        
        # Run evaluation
        result = analytics.comprehensive_evaluation(mock_manager, mock_test_data)
        
        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
        assert 'f1_score' in result
        assert result['sample_count'] == 5
        
        print("✓ Model analytics test passed")
    
    def test_data_quality_analyzer(self):
        """Test data quality analysis"""
        analyzer = DataQualityAnalyzer()
        
        # Analyze test data
        result = analyzer.analyze_data_quality(self.test_data)
        
        assert 'sample_count' in result
        assert 'average_text_length' in result
        assert 'vocabulary_size' in result
        assert 'data_quality_score' in result
        
        # Test insights generation
        insights = analyzer.get_data_insights()
        assert isinstance(insights, list)
        assert len(insights) > 0
        
        print("✓ Data quality analyzer test passed")
    
    def test_memory_optimizer(self):
        """Test memory optimization utilities"""
        optimizer = MemoryOptimizer()
        
        # Test memory clearing (should not raise errors)
        optimizer.clear_memory()
        
        # Test memory stats
        stats = optimizer.get_memory_stats()
        assert isinstance(stats, dict)
        
        # Test context manager
        with optimizer.memory_context():
            # Some operation that uses memory
            _ = [i for i in range(100000)]
        
        print("✓ Memory optimizer test passed")
    
    def test_caching_system(self):
        """Test caching system functionality"""
        cache = CachingSystem(max_size=100)
        
        # Test cache operations
        cache.prediction_cache["test_key"] = "test_value"
        assert "test_key" in cache.prediction_cache
        
        # Test cache clearing
        cache.clear_cache()
        assert len(cache.prediction_cache) == 0
        
        print("✓ Caching system test passed")
    
    def test_background_trainer(self):
        """Test background training system"""
        # Mock model manager
        mock_manager = Mock()
        mock_manager.incremental_fine_tune.return_value = {"status": "success"}
        
        trainer = BackgroundTrainer(mock_manager)
        
        # Test scheduling
        trainer.schedule_training({"data": "test"})
        assert len(trainer.training_queue) == 1
        
        # Test status
        status = trainer.get_training_status()
        assert 'is_training' in status
        assert 'queue_size' in status
        
        print("✓ Background trainer test passed")

def run_phase4_tests():
    """Run all Phase 4 tests"""
    print("🚀 Running Phase 4 Tests...")
    print("=" * 50)
    
    test_suite = TestPhase4()
    
    try:
        test_suite.setup_method()
        test_suite.test_adaptive_learning_rate()
        test_suite.test_smart_sampling()
        test_suite.test_model_analytics()
        test_suite.test_data_quality_analyzer()
        test_suite.test_memory_optimizer()
        test_suite.test_caching_system()
        test_suite.test_background_trainer()
        
        print("=" * 50)
        print("🎉 ALL PHASE 4 TESTS PASSED!")
        print("✓ Adaptive learning rate system working")
        print("✓ Smart sampling for efficient labeling")
        print("✓ Comprehensive model analytics")
        print("✓ Data quality analysis and insights")
        print("✓ Memory optimization utilities")
        print("✓ Caching system for performance")
        print("✓ Background training for non-blocking UI")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_phase4_tests()