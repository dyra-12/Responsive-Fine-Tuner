#!/usr/bin/env python3
"""
Phase 4 Test Runner for Responsive Fine-Tuner
Advanced Features & Deployment Optimizations
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_phase4 import run_phase4_tests

if __name__ == "__main__":
    print("🧪 Responsive Fine-Tuner - Phase 4 Testing")
    print("=" * 60)
    print("Testing: Advanced Features & Deployment Optimizations")
    print("=" * 60)
    
    success = run_phase4_tests()
    
    if success:
        print("\n🎯 Phase 4 completed successfully!")
        print("\n📋 What we've accomplished:")
        print("✅ Adaptive learning rate scheduling")
        print("✅ Smart sampling for efficient labeling")
        print("✅ Reward-based training with TRL")
        print("✅ Comprehensive model analytics")
        print("✅ Data quality analysis and insights")
        print("✅ Memory optimization and caching")
        print("✅ Background training system")
        print("✅ Production deployment setup")
        print("✅ Docker and Nginx configuration")
        
        print("\n🚀 DEPLOYMENT READY!")
        print("\nTo run the advanced application:")
        print("  python run_advanced_app.py")
        print("\nFor production deployment:")
        print("  python run_advanced_app.py --production")
        print("  docker-compose -f deployment/docker-compose.yml up -d")
        
        print("\n🎉 The Responsive Fine-Tuner is now production-ready!")
        print("   With advanced features and optimization for real-world use.")
        
    else:
        print("\n💥 Phase 4 tests failed. Please fix issues before deployment.")
        sys.exit(1)