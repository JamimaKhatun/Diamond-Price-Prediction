#!/usr/bin/env python3
"""
DiamondAI Web Application Launcher
Quick start script for the Diamond Price Predictor web application
"""

import os
import sys
import subprocess
import webbrowser
import time
from threading import Timer

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'flask_cors', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("📦 Installing required dependencies...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'Flask==2.3.3', 'Flask-CORS==4.0.0', 'pandas==2.1.1', 'numpy==1.24.3'
        ])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def open_browser():
    """Open browser after a delay"""
    time.sleep(2)  # Wait for server to start
    webbrowser.open('http://localhost:5000')

def main():
    print("💎 DiamondAI Web Application Launcher")
    print("=" * 50)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        install_choice = input("Install missing dependencies? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_dependencies():
                print("❌ Cannot proceed without dependencies. Exiting.")
                return
        else:
            print("❌ Cannot run without dependencies. Exiting.")
            return
    
    print("✅ All dependencies available!")
    print()
    
    # Check if frontend files exist
    frontend_files = ['frontend/index.html', 'frontend/styles.css', 'frontend/script.js']
    missing_files = [f for f in frontend_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️  Missing frontend files: {', '.join(missing_files)}")
        print("Please ensure all frontend files are in the 'frontend' directory.")
        return
    
    print("🌐 Frontend files found!")
    print()
    
    # Start the application
    print("🚀 Starting DiamondAI Web Application...")
    print("=" * 50)
    print("📱 Application will open at: http://localhost:5000")
    print("🔧 API Health Check: http://localhost:5000/api/health")
    print("=" * 50)
    print("🎯 Demo Login Credentials:")
    print("   Email: demo@diamondai.com")
    print("   Password: demo123")
    print("=" * 50)
    print("💡 Features Available:")
    print("   ✅ User Authentication (Login/Signup)")
    print("   ✅ Diamond Price Prediction")
    print("   ✅ Interactive Dashboard")
    print("   ✅ Responsive Design")
    print("   ✅ Real-time Predictions")
    print("=" * 50)
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Open browser automatically
    Timer(2.0, open_browser).start()
    
    # Import and run the Flask app
    try:
        from app import app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to False for cleaner output
            threaded=True
        )
    except ImportError:
        print("❌ Could not import Flask app. Make sure app.py exists.")
    except KeyboardInterrupt:
        print("\n👋 DiamondAI Web Application stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

if __name__ == "__main__":
    main()