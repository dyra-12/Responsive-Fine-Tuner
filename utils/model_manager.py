import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    PeftModel
)
import numpy as np
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RFTModelManager:
    def __init__(self, base_model_name="distilbert-base-uncased", num_labels=2):
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.model = None
        self.tokenizer = None
        self.lora_config = None
        self.is_initialized = False
        
        # Default label mapping (can be customized by user)
        self.id2label = {0: "Class 0", 1: "Class 1"}
        self.label2id = {"Class 0": 0, "Class 1": 1}
        
        self._setup_device()
        self._load_base_model()
    
    def _setup_device(self):
        """Setup device (GPU if available)"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def _load_base_model(self):
        """Load the base model and tokenizer"""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        self._setup_lora()
        self.model.to(self.device)
        self.is_initialized = True
    
    def _setup_lora(self, r=8, lora_alpha=16, lora_dropout=0.1):
        """Setup LoRA configuration"""
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "v_lin", "k_lin", "out_lin"]  # For DistilBERT
        )
        
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
    
    def update_label_mapping(self, id2label):
        """Update label mapping based on user input"""
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
        self.num_labels = len(id2label)
    
    def predict(self, texts):
        """Predict labels for a list of texts"""
        if not self.is_initialized:
            raise ValueError("Model not initialized")
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize inputs
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_ids = predictions.argmax(dim=-1).cpu().numpy()
            confidence_scores = predictions.max(dim=-1).values.cpu().numpy()
        
        # Convert to labels
        predicted_labels = [self.id2label[class_id] for class_id in predicted_class_ids]
        
        return predicted_labels, confidence_scores.tolist()
    
    def train_on_feedback(self, feedback_dataset, learning_rate=5e-4, num_epochs=3):
        """Fine-tune the model on user feedback"""
        if len(feedback_dataset) == 0:
            logger.warning("No feedback data to train on")
            return
        
        logger.info(f"Training on {len(feedback_dataset)} feedback samples")
        
        # Prepare dataset
        texts = [item['text'] for item in feedback_dataset]
        labels = [self.label2id[item['label']] for item in feedback_dataset]
        
        # Tokenize
        tokenized_data = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        
        # Create torch dataset
        class FeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = FeedbackDataset(tokenized_data, labels)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./lora_adapters",
            learning_rate=learning_rate,
            per_device_train_batch_size=4,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",
            remove_unused_columns=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        logger.info("Fine-tuning completed")
    
    def evaluate(self, test_texts, test_labels):
        """Evaluate model on test set"""
        predicted_labels, confidence_scores = self.predict(test_texts)
        
        # Convert string labels to IDs for comparison
        true_label_ids = [self.label2id[label] for label in test_labels]
        pred_label_ids = [self.label2id[label] for label in predicted_labels]
        
        accuracy = np.mean(np.array(true_label_ids) == np.array(pred_label_ids))
        
        return accuracy, predicted_labels, confidence_scores