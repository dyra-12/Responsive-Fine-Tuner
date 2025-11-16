import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig
from trl import RewardTrainer, RewardConfig

from backend.enhanced_model_manager import EnhancedModelManager, FeedbackDataset
from backend.data_processor import ProcessedData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveLearningRate:
    """Adaptive learning rate scheduler based on feedback quality"""
    
    def __init__(self, initial_lr: float = 1e-4, min_lr: float = 1e-6, max_lr: float = 1e-3):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_lr = initial_lr
        self.accuracy_history = deque(maxlen=10)
        self.loss_history = deque(maxlen=10)
        
    def update(self, accuracy: float, loss: float) -> float:
        """Update learning rate based on performance"""
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        
        if len(self.accuracy_history) < 3:
            return self.current_lr

        # Calculate trends. If we don't have enough history for two 3-point windows,
        # fall back to a simple first-to-last comparison to determine trend.
        if len(self.accuracy_history) < 6:
            accuracy_trend = self.accuracy_history[-1] - self.accuracy_history[0]
            loss_trend = self.loss_history[-1] - self.loss_history[0]
        else:
            accuracy_trend = np.mean(list(self.accuracy_history)[-3:]) - np.mean(list(self.accuracy_history)[-6:-3])
            loss_trend = np.mean(list(self.loss_history)[-3:]) - np.mean(list(self.loss_history)[-6:-3])
        
        # Adjust learning rate
        if accuracy_trend > 0.02 and loss_trend < -0.02:  # Improving
            self.current_lr = min(self.current_lr * 1.2, self.max_lr)
        elif accuracy_trend < -0.01 or loss_trend > 0.01:  # Worsening
            self.current_lr = max(self.current_lr * 0.8, self.min_lr)
        
        logger.info(f"Learning rate adjusted to: {self.current_lr:.2e}")
        return self.current_lr

class SmartSampling:
    """Smart sampling for selecting most informative samples"""
    
    def __init__(self, strategy: str = "uncertainty"):
        self.strategy = strategy
        self.sample_uncertainties = {}
        
    def calculate_uncertainty(self, probabilities: List[float]) -> float:
        """Calculate uncertainty using entropy"""
        probabilities = np.array(probabilities)
        return -np.sum(probabilities * np.log(probabilities + 1e-8))
    
    def update_sample_uncertainty(self, text: str, probabilities: List[float]):
        """Update uncertainty for a sample"""
        uncertainty = self.calculate_uncertainty(probabilities)
        self.sample_uncertainties[text] = uncertainty
    
    def get_most_uncertain_samples(self, texts: List[str], n: int = 5) -> List[str]:
        """Get the n most uncertain samples"""
        if not self.sample_uncertainties:
            return texts[:n]
        
        # Get uncertainties for available texts
        uncertainties = []
        for text in texts:
            if text in self.sample_uncertainties:
                uncertainties.append((text, self.sample_uncertainties[text]))
            else:
                uncertainties.append((text, 0.5))  # Default medium uncertainty
        
        # Sort by uncertainty (descending)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        return [text for text, _ in uncertainties[:n]]

class RewardBasedTrainer:
    """Reward-based training using TRL for preference learning"""
    
    def __init__(self, model_manager: EnhancedModelManager):
        self.model_manager = model_manager
        self.reward_history = []
        
    def compute_reward(self, prediction: Dict, user_feedback: str, correct_label: int) -> float:
        """Compute reward based on prediction quality and user feedback"""
        predicted_label = prediction.get('predicted_label', -1)
        confidence = prediction.get('confidence', 0.0)
        
        if user_feedback == "Correct":
            # Reward correct predictions with higher confidence
            base_reward = 1.0
            confidence_bonus = confidence * 0.5
            return base_reward + confidence_bonus
        else:
            # Penalize incorrect predictions, but less if model was uncertain
            base_penalty = -1.0
            uncertainty_bonus = (1 - confidence) * 0.5  # Less penalty if uncertain
            return base_penalty + uncertainty_bonus
    
    def prepare_reward_data(self, feedback_data: List[Dict]) -> List[Dict]:
        """Prepare data for reward training"""
        reward_samples = []
        
        for feedback in feedback_data:
            text = feedback['text']
            user_label = feedback['user_label']
            model_pred = feedback['model_prediction']
            
            reward = self.compute_reward(model_pred, 
                                       "Correct" if model_pred['predicted_label'] == user_label else "Incorrect",
                                       user_label)
            
            reward_samples.append({
                'text': text,
                'reward': reward,
                'user_label': user_label
            })
        
        return reward_samples

class AdvancedModelManager(EnhancedModelManager):
    """Enhanced model manager with advanced training features"""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.adaptive_lr = AdaptiveLearningRate(initial_lr=self.config.training.learning_rate)
        self.smart_sampler = SmartSampling()
        self.reward_trainer = RewardBasedTrainer(self)
        self.training_metrics_history = []
        
    def fine_tune_with_adaptive_lr(self, processed_data: ProcessedData, epochs: int = 1) -> Dict[str, Any]:
        """Fine-tune with adaptive learning rate"""
        try:
            # Prepare dataset
            dataset = self.prepare_dataset(processed_data)
            
            if len(dataset) == 0:
                return {"status": "no_data"}
            
            # Update learning rate based on recent performance
            recent_accuracy = 0.5  # Default
            recent_loss = 1.0      # Default
            if self.training_metrics_history:
                recent_metrics = self.training_metrics_history[-1]
                recent_accuracy = recent_metrics.get('accuracy', 0.5)
                recent_loss = recent_metrics.get('loss', 1.0)
            
            current_lr = self.adaptive_lr.update(recent_accuracy, recent_loss)
            
            # Training arguments with adaptive LR
            training_args = TrainingArguments(
                output_dir="./training_output",
                overwrite_output_dir=True,
                per_device_train_batch_size=self.config.training.batch_size,
                num_train_epochs=epochs,
                max_steps=min(self.config.training.max_steps, len(dataset) * epochs),
                learning_rate=current_lr,
                warmup_steps=min(10, len(dataset) // 2),
                logging_steps=5,
                save_steps=100,
                evaluation_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,
                remove_unused_columns=False,
                report_to=None,
                dataloader_pin_memory=False,
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding=True,
                max_length=self.config.model.max_length,
            )
            
            # Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # Train
            logger.info(f"Starting adaptive training with LR: {current_lr:.2e}")
            train_result = trainer.train()
            
            # Store metrics
            training_metrics = {
                'epoch': epochs,
                'training_samples': len(dataset),
                'train_loss': train_result.training_loss,
                'learning_rate': current_lr,
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_metrics_history.append(training_metrics)
            
            return {
                "status": "success",
                "loss": train_result.training_loss,
                "learning_rate": current_lr,
                "samples_trained": len(dataset)
            }
            
        except Exception as e:
            logger.error(f"Adaptive training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_smart_samples(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get smartly sampled documents for labeling"""
        if not self.train_data:
            return []
        
        # Use smart sampling to select most uncertain samples
        selected_texts = self.smart_sampler.get_most_uncertain_samples(
            self.train_data.texts, n
        )
        
        samples = []
        for text in selected_texts:
            prediction = self.get_prediction(text)
            samples.append({
                'text': text,
                'prediction': prediction,
                'metadata': next((m for m in self.train_data.metadata if m.get('text') == text), {})
            })
            
            # Update uncertainty for future sampling
            self.smart_sampler.update_sample_uncertainty(
                text, prediction.get('probabilities', [])
            )
        
        return samples
    
    def reward_based_training(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Perform reward-based training using TRL"""
        try:
            if not feedback_data:
                return {"status": "no_data"}
            
            # Prepare reward data
            reward_samples = self.reward_trainer.prepare_reward_data(feedback_data)
            
            if len(reward_samples) < 2:
                return {"status": "insufficient_data"}
            
            # Convert to dataset format (simplified - actual TRL implementation would be more complex)
            texts = [sample['text'] for sample in reward_samples]
            rewards = [sample['reward'] for sample in reward_samples]
            
            # For now, use standard fine-tuning with reward-weighted sampling
            # In a full implementation, we would use TRL's RewardTrainer
            processed_data = ProcessedData(
                texts=texts,
                labels=[0] * len(texts),  # Placeholder - rewards would be used differently
                metadata=reward_samples,
                file_sources=["reward_training"]
            )
            
            result = self.fine_tune_with_adaptive_lr(processed_data, epochs=1)
            
            if result["status"] == "success":
                logger.info(f"Reward-based training completed with {len(reward_samples)} samples")
                self.reward_trainer.reward_history.extend(reward_samples)
            
            return result
            
        except Exception as e:
            logger.error(f"Reward-based training failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_training_analytics(self) -> Dict[str, Any]:
        """Get comprehensive training analytics"""
        base_info = self.get_model_info()
        
        analytics = {
            **base_info,
            'adaptive_learning_rate': self.adaptive_lr.current_lr,
            'training_sessions_count': len(self.training_metrics_history),
            'reward_training_samples': len(self.reward_trainer.reward_history),
            'smart_sampling_enabled': True,
            'performance_trend': self._calculate_performance_trend()
        }
        
        if self.training_metrics_history:
            recent_training = self.training_metrics_history[-1]
            analytics.update({
                'recent_training_loss': recent_training.get('train_loss'),
                'recent_learning_rate': recent_training.get('learning_rate'),
                'recent_training_samples': recent_training.get('training_samples')
            })
        
        return analytics
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend based on recent metrics"""
        if len(self.training_metrics_history) < 2:
            return "stable"
        
        recent_losses = [m.get('train_loss', 1.0) for m in self.training_metrics_history[-3:]]
        if len(recent_losses) < 2:
            return "stable"
        
        loss_trend = np.mean(recent_losses[:-1]) - recent_losses[-1]
        
        if loss_trend > 0.1:
            return "improving"
        elif loss_trend < -0.1:
            return "worsening"
        else:
            return "stable"