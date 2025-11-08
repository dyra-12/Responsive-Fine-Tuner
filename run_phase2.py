#!/usr/bin/env python3
"""
Phase 2 Test Runner for Responsive Fine-Tuner
Data Processing & Training Pipeline
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_phase2 import run_phase2_tests

if __name__ == "__main__":
    print("🧪 Responsive Fine-Tuner - Phase 2 Testing")
    print("=" * 60)
    print("Testing: Data Processing & Training Pipeline")
    print("=" * 60)
    
    success = run_phase2_tests()
    
    if success:
        print("\n🎯 Phase 2 completed successfully!")
        print("\n📋 What we've accomplished:")
        print("✅ Data processing pipeline for TXT and CSV files")
        print("✅ File validation and encoding detection")
        print("✅ Train-test splitting functionality")
        print("✅ Enhanced model manager with training capabilities")
        print("✅ LoRA fine-tuning implementation")
        print("✅ Model evaluation system")
        print("✅ Data persistence (save/load)")
        print("✅ Comprehensive testing framework")
        
        print("\n🚀 Ready for Phase 3: Frontend Development!")
        print("Next: Building the Gradio interface and interactive components")
    else:
        print("\n💥 Phase 2 tests failed. Please fix issues before proceeding.")
        sys.exit(1)