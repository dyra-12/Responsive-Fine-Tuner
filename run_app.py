#!/usr/bin/env python3
"""
Main launcher for Responsive Fine-Tuner
"""

import os
import sys
import argparse

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.main_app import RFTInterface

def main():
    """Main application launcher"""
    parser = argparse.ArgumentParser(description="Responsive Fine-Tuner")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config file")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    
    args = parser.parse_args()
    
    print("🎯 Responsive Fine-Tuner")
    print("=" * 50)
    print("Starting application...")
    
    try:
        # Create and launch interface
        interface = RFTInterface(args.config)
        
        launch_params = {
            "server_port": args.port,
            "share": args.share,
            "debug": args.debug
        }
        
        print(f"🚀 Launching on port {args.port}...")
        print(f"📊 Debug mode: {args.debug}")
        print(f"🌐 Public sharing: {args.share}")
        print("\nAccess the application at: http://localhost:7860")
        print("Press Ctrl+C to stop the application")
        
        interface.launch(**launch_params)
        
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()