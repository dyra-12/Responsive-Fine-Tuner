import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackLoop:
    def __init__(self):
        self.feedback_data = []
        self.feedback_count = 0
    
    def add_feedback(self, text, model_label, user_label):
        """Add a user correction to the feedback dataset"""
        feedback_item = {
            'text': text,
            'model_label': model_label,
            'user_label': user_label
        }
        
        # Only add if it's a correction (user_label != model_label)
        if user_label != model_label:
            self.feedback_data.append(feedback_item)
            self.feedback_count += 1
            logger.info(f"Added feedback: '{model_label}' -> '{user_label}'")
            return True
        return False
    
    def get_feedback_count(self):
        """Get the number of feedback samples"""
        return self.feedback_count
    
    def get_feedback_for_training(self):
        """Get feedback data in training format"""
        training_data = []
        for item in self.feedback_data:
            training_data.append({
                'text': item['text'],
                'label': item['user_label']  # Use the corrected label for training
            })
        return training_data
    
    def clear_feedback(self):
        """Clear all feedback data"""
        self.feedback_data = []
        self.feedback_count = 0
        logger.info("Feedback data cleared")
    
    def get_feedback_summary(self):
        """Get a summary of the feedback data"""
        if not self.feedback_data:
            return "No feedback collected yet"
        
        df = pd.DataFrame(self.feedback_data)
        corrections = len(df[df['model_label'] != df['user_label']])
        
        return f"Total feedback: {self.feedback_count} | Corrections: {corrections}"