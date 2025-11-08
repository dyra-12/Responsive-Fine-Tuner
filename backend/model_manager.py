import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    pipeline
)
from peft import LoraConfig, get_peft_model
import logging
from typing import Dict, List, Any
import datetime
import numpy as np

from backend.config import ConfigManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, config: ConfigManager = None):
        self.config = config or ConfigManager()
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        self.is_initialized = False
        self.feedback_data = []
        
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the base model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.config.model.base_model}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.base_model
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.base_model,
                num_labels=self.config.model.num_labels
            )
            
            # Setup LoRA configuration
            self.setup_lora()
            
            self.is_initialized = True
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        try:
            # Determine sensible default target modules for LoRA.
            # PEFT expects a list of module name substrings to match linear projection modules.
            # Try common attention projection names first; if none are found, fall back to
            # the classifier head or the first nn.Linear module we can find.
            import torch.nn as nn

            candidates = [
                "q_proj", "v_proj", "k_proj",
                "q_lin", "v_lin", "k_lin",
                "query", "value", "key",
                "q", "v", "k",
            ]

            available_module_names = [name for name, _ in self.model.named_modules()]

            matched = []
            for cand in candidates:
                for name in available_module_names:
                    if cand in name:
                        matched.append(cand)
                        break

            # Remove duplicates while preserving order
            matched = list(dict.fromkeys(matched))

            # If nothing matched, try the classifier head
            if not matched:
                if "classifier" in available_module_names:
                    matched = ["classifier"]
                else:
                    # As a last resort, find the first Linear module and use its attribute name
                    first_linear = None
                    for name, module in self.model.named_modules():
                        if isinstance(module, nn.Linear):
                            first_linear = name.split(".")[-1]
                            break
                    if first_linear:
                        matched = [first_linear]

            if not matched:
                raise RuntimeError(
                    "No suitable target modules found for LoRA. "
                    "Please set `training.target_modules` in your config or ensure the model contains identifiable projection modules."
                )

            logger.info(f"Using LoRA target modules: {matched}")

            self.lora_config = LoraConfig(
                r=self.config.training.lora_r,
                lora_alpha=self.config.training.lora_alpha,
                target_modules=matched,
                lora_dropout=self.config.training.lora_dropout,
                bias="none",
                task_type="SEQ_CLS"
            )

            # Apply LoRA to model
            self.model = get_peft_model(self.model, self.lora_config)
            logger.info("LoRA configuration applied successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup LoRA: {e}")
            raise
    
    def get_prediction(self, text: str) -> Dict[str, Any]:
        """Get model prediction for a single text"""
        if not self.is_initialized:
            raise ValueError("Model not initialized")
        
        try:
            # Treat empty or whitespace-only strings as invalid input and return an error
            if text is None or (isinstance(text, str) and text.strip() == ""):
                return {
                    "text": text,
                    "error": "empty input",
                    "predicted_label": -1,
                    "probabilities": [],
                    "confidence": 0.0
                }

            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.config.model.max_length,
                padding=True
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Convert to probabilities and labels
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
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get predictions for multiple texts"""
        return [self.get_prediction(text) for text in texts]
    
    def add_feedback(self, text: str, user_label: int, model_prediction: Dict):
        """Store user feedback for training"""
        feedback_entry = {
            "text": text,
            "user_label": user_label,
            "model_prediction": model_prediction,
            # Store timestamp as an ISO-formatted string to avoid numpy.datetime64 -> torch.tensor
            # conversion issues. Tests and other components only need a retrievable timestamp.
            "timestamp": datetime.datetime.utcnow().isoformat()
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
            "status": "initialized"
        }