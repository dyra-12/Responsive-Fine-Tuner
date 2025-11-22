#!/usr/bin/env python3
"""
Phase 5 Test Runner for Responsive Fine-Tuner
Enterprise Features & Scalability
"""

import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tests.test_phase5 import run_phase5_tests

if __name__ == "__main__":
    print("🧪 Responsive Fine-Tuner - Phase 5 Testing")
    print("=" * 60)
    print("Testing: Enterprise Features & Scalability")
    print("=" * 60)
    
    success = run_phase5_tests()
    
    if success:
        print("\n🎯 Phase 5 completed successfully!")
        print("\n📋 What we've accomplished:")
        print("✅ Multi-user authentication system")
        print("✅ JWT-based stateless authentication")
        print("✅ Project management with data isolation")
        print("✅ Role-based access control (RBAC)")
        print("✅ Enterprise application core")
        print("✅ User session management")
        print("✅ Security and audit logging")
        print("✅ Enterprise frontend with user management")
        print("✅ Scalable deployment architecture")
        
        print("\n🚀 ENTERPRISE READY!")
        print("\nTo run the enterprise application:")
        print("  python run_enterprise.py")
        print("\nFor production deployment:")
        print("  python run_enterprise.py --production")
        print("  docker-compose -f deployment/enterprise-docker-compose.yml up -d")
        
        print("\n🎉 The Responsive Fine-Tuner is now ENTERPRISE-READY!")
        print("   With multi-user support, security, and scalability for organizational use.")
        
    else:
        print("\n💥 Phase 5 tests failed. Please fix issues before enterprise deployment.")
        sys.exit(1)