#!/usr/bin/env python3
"""
Enterprise launcher for Responsive Fine-Tuner
"""

import os
import sys
import argparse

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.enterprise_interface import EnterpriseRFTInterface

def main():
    """Main enterprise application launcher"""
    parser = argparse.ArgumentParser(description="Enterprise Responsive Fine-Tuner")
    parser.add_argument("--config", default="config/enterprise.yaml", help="Path to config file")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--production", action="store_true", help="Enable production mode")
    parser.add_argument("--max-users", type=int, default=100, help="Maximum number of users")
    
    args = parser.parse_args()
    
    print("🏢 Enterprise Responsive Fine-Tuner")
    print("=" * 50)
    print("Starting enterprise application...")
    
    try:
        # Create and launch enterprise interface
        interface = EnterpriseRFTInterface(args.config)
        
        launch_params = {
            "server_port": args.port,
            "share": args.share,
            "debug": args.debug
        }
        
        if args.production:
            print("🏭 Production mode enabled")
            launch_params["show_error"] = True
        
        print(f"🚀 Launching on port {args.port}...")
        print(f"📊 Debug mode: {args.debug}")
        print(f"🌐 Public sharing: {args.share}")
        print(f"🏭 Production mode: {args.production}")
        print(f"👥 Max users: {args.max_users}")
        print("\nAccess the application at: http://localhost:7860")
        print("Press Ctrl+C to stop the application")
        
        interface.launch(**launch_params)
        
    except Exception as e:
        print(f"❌ Failed to launch enterprise application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()