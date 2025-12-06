#!/usr/bin/env python3
"""
Final comprehensive test suite for Responsive Fine-Tuner
"""

import os
import sys
import subprocess
from datetime import datetime

def run_all_tests():
    """Run all test suites in order"""
    print("🧪 COMPREHENSIVE TEST SUITE - Responsive Fine-Tuner")
    print("=" * 70)
    print("Running all test phases...")
    print("=" * 70)
    
    test_suites = [
        ("Phase 1: Foundation", "run_phase1.py"),
        ("Phase 2: Core Architecture", "run_phase2.py"),
        ("Phase 3: Frontend Development", "run_phase3.py"),
        ("Phase 4: Advanced Features", "run_phase4.py"),
        ("Phase 5: Enterprise Features", "run_phase5.py"),
        ("Phase 6: Production Deployment", "run_phase6.py")
    ]
    
    results = []
    
    for phase_name, test_file in test_suites:
        print(f"\n▶️  Testing {phase_name}...")
        print("-" * 50)
        
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(__file__)
            )
            
            success = result.returncode == 0
            results.append((phase_name, success, result.stdout))
            
            if success:
                print(f"✅ {phase_name}: PASSED")
            else:
                print(f"❌ {phase_name}: FAILED")
                print(f"Error output:\n{result.stderr}")
                
        except Exception as e:
            print(f"❌ {phase_name}: ERROR - {e}")
            results.append((phase_name, False, str(e)))
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for phase_name, success, output in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {phase_name}")
    
    print(f"\n🎯 Overall: {passed}/{total} phases passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The system is ready for production.")
        print("\nTo deploy the complete system:")
        print("  python run_production.py")
        print("\nFor enterprise deployment:")
        print("  python run_enterprise.py --production")
        print("\nFor development:")
        print("  python run_app.py")
        return True
    else:
        print(f"\n⚠️  {total - passed} test suites failed. Please fix issues.")
        return False

if __name__ == "__main__":
    start_time = datetime.now()
    success = run_all_tests()
    end_time = datetime.now()
    
    print(f"\n⏱️  Total testing time: {(end_time - start_time).total_seconds():.1f} seconds")
    sys.exit(0 if success else 1)