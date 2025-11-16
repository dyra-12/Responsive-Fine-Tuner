#!/usr/bin/env python3
"""
Advanced application launcher with all Phase 4 features
"""

import os
import sys
import argparse

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.main_app import RFTInterface
from backend.advanced_trainer import AdvancedModelManager
from backend.analytics import ModelAnalytics, DataQualityAnalyzer
from backend.optimizations import MemoryOptimizer, BackgroundTrainer

class AdvancedRFTInterface(RFTInterface):
    """Enhanced RFT interface with Phase 4 features"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        super().__init__(config_path)
        
        # Replace with advanced components
        self.app.model_manager = AdvancedModelManager(self.app.config)
        self.app.analytics = ModelAnalytics()
        self.app.data_analyzer = DataQualityAnalyzer()
        self.app.memory_optimizer = MemoryOptimizer()
        self.app.background_trainer = BackgroundTrainer(self.app.model_manager)
        
        print("🚀 Advanced Responsive Fine-Tuner initialized!")
        print("📊 Features enabled:")
        print("  • Adaptive learning rate")
        print("  • Smart sampling")
        print("  • Comprehensive analytics")
        print("  • Data quality insights")
        print("  • Memory optimization")
        print("  • Background training")

def main():
    """Main advanced application launcher"""
    parser = argparse.ArgumentParser(description="Advanced Responsive Fine-Tuner")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--production", action="store_true", help="Enable production mode")
    
    args = parser.parse_args()
    
    print("🎯 Advanced Responsive Fine-Tuner")
    print("=" * 50)
    print("Starting advanced application...")
    
    try:
        # Create and launch advanced interface
        interface = AdvancedRFTInterface(args.config)
        
        launch_params = {
            "server_port": args.port,
            "share": args.share,
            "debug": args.debug
        }
        
        if args.production:
            print("🏭 Production mode enabled")
            # Additional production settings
            launch_params["show_error"] = True
            # Use the current Gradio parameter name for queuing. Older/newer
            # versions may use 'queue' instead of 'enable_queue'. Use 'queue'
            # which is accepted by most versions of Gradio's Blocks.launch.
            launch_params["queue"] = True
        
        print(f"🚀 Launching on port {args.port}...")
        print(f"📊 Debug mode: {args.debug}")
        print(f"🌐 Public sharing: {args.share}")
        print(f"🏭 Production mode: {args.production}")
        print("\nAccess the application at: http://localhost:7860")
        print("Press Ctrl+C to stop the application")
        
        interface.launch(**launch_params)
        
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()