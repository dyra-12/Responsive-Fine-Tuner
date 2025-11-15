#!/usr/bin/env python3
"""
Phase 3 Test Runner for Responsive Fine-Tuner
Frontend Development & Interactive Interface
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_phase3 import run_phase3_tests

if __name__ == "__main__":
    print("🧪 Responsive Fine-Tuner - Phase 3 Testing")
    print("=" * 60)
    print("Testing: Frontend Development & Interactive Interface")
    print("=" * 60)
    
    success = run_phase3_tests()
    
    if success:
        print("\n🎯 Phase 3 completed successfully!")
        print("\n📋 What we've accomplished:")
        print("✅ Complete Gradio frontend interface")
        print("✅ Data upload and processing workflow")
        print("✅ Interactive labeling with real-time feedback")
        print("✅ Performance monitoring dashboard")
        print("✅ Model prediction visualization")
        print("✅ Progress tracking and metrics")
        print("✅ Settings and configuration management")
        print("✅ Comprehensive application state management")
        
        print("\n🚀 Ready to launch the full application!")
        print("\nTo run the application:")
        print("  python run_app.py")
        print("\nOptional arguments:")
        print("  --share     Create public share link")
        print("  --debug     Enable debug mode") 
        print("  --port X    Run on specific port (default: 7860)")
        
    else:
        print("\n💥 Phase 3 tests failed. Please fix issues before proceeding.")
        sys.exit(1)