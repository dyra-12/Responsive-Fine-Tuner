#!/usr/bin/env python3
"""
Phase 1 Test Runner for Responsive Fine-Tuner
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_phase1 import run_phase1_tests

if __name__ == "__main__":
    print("🧪 Responsive Fine-Tuner - Phase 1 Testing")
    print("=" * 60)
    
    success = run_phase1_tests()
    
    if success:
        print("\n🎯 Phase 1 completed successfully!")
        print("Next steps:")
        print("1. Run 'python run_phase1.py' to verify everything works")
        print("2. Proceed to Phase 2: Data processing and training pipeline")
        print("3. The foundation is solid - ready to build interactive features")
    else:
        print("\n💥 Phase 1 tests failed. Please fix issues before proceeding.")
        sys.exit(1)