import yaml
import os
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    base_model: str
    max_length: int
    num_labels: int

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float

@dataclass
class DataConfig:
    supported_formats: List[str]
    max_file_size_mb: int
    test_split: float

@dataclass
class UIConfig:
    theme: str
    batch_size_labeling: int
    update_interval: int

class ConfigManager:
    def __init__(self, config_path="config/settings.yaml"):
        self.config_path = config_path
        self.load_config()
    
    def load_config(self):
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        self.model = ModelConfig(**config_data['model'])
        self.training = TrainingConfig(**config_data['training'])
        self.data = DataConfig(**config_data['data'])
        self.ui = UIConfig(**config_data['ui'])
    
    def get_config(self):
        return {
            'model': self.model,
            'training': self.training,
            'data': self.data,
            'ui': self.ui
        }