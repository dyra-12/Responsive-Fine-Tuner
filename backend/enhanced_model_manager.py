import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime
import os

from backend.config import ConfigManager
from backend.data_processor import ProcessedData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackDataset(Dataset):
    """Dataset for feedback-based training"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EnhancedModelManager:
    def __init__(self, config: ConfigManager = None):
        self.config = config or ConfigManager()
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        self.is_initialized = False
        self.feedback_data = []
        self.training_history = []
        self.current_dataset = None
        
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the base model and tokenizer with enhanced error handling"""
        try:
            logger.info(f"Loading model: {self.config.model.base_model}")
            
            # Load tokenizer with special settings
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.base_model,
                use_fast=True
            )
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.base_model,
                num_labels=self.config.model.num_labels,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            # Resize token embeddings if needed
            if self.tokenizer.vocab_size != self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Setup LoRA
            self.setup_lora()
            
            self.is_initialized = True
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning with enhanced settings"""
        try:
            # Different target modules for different model architectures
            if "distilbert" in self.config.model.base_model.lower():
                target_modules = ["q_lin", "v_lin", "pre_classifier"]
            elif "bert" in self.config.model.base_model.lower():
                target_modules = ["query", "value", "intermediate.dense"]
            else:
                target_modules = ["q_proj", "v_proj", "down_proj"]
            
            # Use string task_type to avoid version-specific enum handling in PEFT
            self.lora_config = LoraConfig(
                r=self.config.training.lora_r,
                lora_alpha=self.config.training.lora_alpha,
                target_modules=target_modules,
                lora_dropout=self.config.training.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
            )
            
            # Apply LoRA to model. Some PEFT versions or model combinations can raise
            # TypeError during adapter injection (e.g., incompatible `modules_to_save`).
            # In that case, log and continue without LoRA so the enhanced manager
            # remains usable in constrained environments (tests can proceed).
            try:
                self.model = get_peft_model(self.model, self.lora_config)
                # Print trainable parameters
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"LoRA applied: {trainable_params}/{total_params} parameters trainable ({trainable_params/total_params*100:.2f}%)")
            except TypeError as te:
                logger.warning(f"LoRA not applied due to TypeError: {te}")
                self.lora_config = None
                # Model remains the base model without PEFT wrappers
                return
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            raise
    
    def prepare_dataset(self, processed_data: ProcessedData):
        """Prepare dataset for training"""
        if not self.is_initialized:
            raise ValueError("Model not initialized")
        
        try:
            dataset = FeedbackDataset(
                texts=processed_data.texts,
                labels=processed_data.labels,
                tokenizer=self.tokenizer,
                max_length=self.config.model.max_length
            )
            
            self.current_dataset = dataset
            logger.info(f"Dataset prepared with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            raise
    
    def fine_tune_step(self, processed_data: ProcessedData, epochs: int = 1):
        """Single fine-tuning step with the provided data"""
        if not self.is_initialized:
            raise ValueError("Model not initialized")
        
        try:
            # Prepare dataset
            dataset = self.prepare_dataset(processed_data)
            
            if len(dataset) == 0:
                logger.warning("No data available for training")
                return {"status": "no_data"}
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./training_output",
                overwrite_output_dir=True,
                per_device_train_batch_size=self.config.training.batch_size,
                per_device_eval_batch_size=self.config.training.batch_size,
                num_train_epochs=epochs,
                max_steps=min(self.config.training.max_steps, len(dataset) * epochs),
                learning_rate=self.config.training.learning_rate,
                warmup_steps=min(10, len(dataset) // 2),
                logging_steps=5,
                save_steps=100,
                evaluation_strategy="no",
                save_strategy="no",
                load_best_model_at_end=False,
                remove_unused_columns=False,
                report_to=None,  # Disable wandb/tensorboard
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
            logger.info(f"Starting training with {len(dataset)} samples for {epochs} epochs")
            train_result = trainer.train()
            
            # Log training results
            training_metrics = {
                'epoch': epochs,
                'training_samples': len(dataset),
                'train_loss': train_result.training_loss,
                'train_metrics': train_result.metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            self.training_history.append(training_metrics)
            logger.info(f"Training completed. Loss: {train_result.training_loss:.4f}")
            
            return {
                "status": "success",
                "loss": train_result.training_loss,
                "samples_trained": len(dataset),
                "epochs": epochs
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def incremental_fine_tune(self, new_feedback_data: List[Dict]):
        """Incremental fine-tuning with new feedback data"""
        if not new_feedback_data:
            logger.warning("No new feedback data for incremental training")
            return {"status": "no_new_data"}
        
        try:
            # Extract texts and labels from feedback
            texts = [fb['text'] for fb in new_feedback_data]
            labels = [fb['user_label'] for fb in new_feedback_data]
            
            # Create processed data object
            processed_data = ProcessedData(
                texts=texts,
                labels=labels,
                metadata=new_feedback_data,
                file_sources=["feedback"]
            )
            
            # Fine-tune with a single epoch for quick updates
            result = self.fine_tune_step(processed_data, epochs=1)
            
            if result["status"] == "success":
                logger.info(f"Incremental training completed with {len(new_feedback_data)} samples")
                # Clear the feedback data that was used for training
                self.feedback_data = [fb for fb in self.feedback_data if fb not in new_feedback_data]
            
            return result
            
        except Exception as e:
            logger.error(f"Incremental training failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def evaluate_model(self, test_data: ProcessedData) -> Dict[str, Any]:
        """Evaluate model on test data"""
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        try:
            self.model.eval()
            predictions = []
            actual_labels = []
            
            with torch.no_grad():
                for text, true_label in zip(test_data.texts, test_data.labels):
                    pred = self.get_prediction(text)
                    predictions.append(pred['predicted_label'])
                    actual_labels.append(true_label)
            
            # Calculate accuracy
            correct = sum(1 for p, a in zip(predictions, actual_labels) if p == a)
            accuracy = correct / len(predictions) if predictions else 0
            
            # Calculate confidence statistics
            confidences = [self.get_prediction(text)['confidence'] for text in test_data.texts]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            results = {
                "accuracy": accuracy,
                "total_samples": len(predictions),
                "correct_predictions": correct,
                "average_confidence": avg_confidence,
                "predictions": predictions,
                "actual_labels": actual_labels
            }
            
            logger.info(f"Evaluation completed: Accuracy = {accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"error": str(e)}
    
    def get_training_history(self) -> List[Dict]:
        """Get the training history"""
        return self.training_history
    
    def save_model(self, save_path: str):
        """Save the current model state"""
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            # Save training history
            history_path = os.path.join(save_path, "training_history.json")
            import json
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2, default=str)
            
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    # Inherit prediction methods from base ModelManager
    def get_prediction(self, text: str) -> Dict[str, Any]:
        """Get model prediction for a single text"""
        if not self.is_initialized:
            return {"error": "Model not initialized"}
        
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.model.max_length,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions[0].cpu().numpy()
            predicted_label = int(torch.argmax(predictions, dim=-1)[0])
            
            return {
                "text": text,
                "predicted_label": predicted_label,
                "probabilities": probs.tolist(),
                "confidence": float(np.max(probs))
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                "text": text,
                "error": str(e),
                "predicted_label": -1,
                "probabilities": [],
                "confidence": 0.0
            }
    
    def add_feedback(self, text: str, user_label: int, model_prediction: Dict):
        """Store user feedback for training"""
        feedback_entry = {
            "text": text,
            "user_label": user_label,
            "model_prediction": model_prediction,
            "timestamp": datetime.now().isoformat()
        }
        self.feedback_data.append(feedback_entry)
        logger.info(f"Feedback added. Total feedback samples: {len(self.feedback_data)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "base_model": self.config.model.base_model,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": (trainable_params / total_params) * 100,
            "lora_enabled": self.lora_config is not None,
            "feedback_samples": len(self.feedback_data),
            "training_sessions": len(self.training_history),
            "status": "initialized"
        }