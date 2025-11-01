#!/usr/bin/env python3
"""
Test script to verify the project setup
"""

import sys
import os

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

try:
    from utils.data_manager import DataManager
    from utils.model_manager import RFTModelManager
    from utils.feedback_loop import FeedbackLoop
    print("✅ All imports successful!")
    
    # Test data manager
    dm = DataManager()
    print("✅ DataManager initialized")
    
    # Test feedback loop
    fl = FeedbackLoop()
    print("✅ FeedbackLoop initialized")
    
    # Test model manager (this might take a moment)
    print("Testing ModelManager...")
    mm = RFTModelManager()
    print("✅ ModelManager initialized")
    
    # Test prediction
    test_texts = ["This is great!", "This is terrible!"]
    labels, confidences = mm.predict(test_texts)
    print(f"✅ Predictions working: {labels}")
    
    print("\n🎉 All systems go! Project setup is complete.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)