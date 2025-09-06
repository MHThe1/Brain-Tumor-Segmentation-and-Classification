#!/usr/bin/env python3
"""
Launch Jupyter Notebook for Brain Tumor Analysis
BRACU CSE428 Academic Project
"""

import os
import sys
import subprocess
import webbrowser
import time

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"📓 {title}")
    print("="*60)

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check if we're in the right directory
    if not os.path.exists("brain_tumor_analysis.ipynb"):
        print("❌ brain_tumor_analysis.ipynb not found in current directory")
        return False
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment may not be activated")
        print("   Please run: source venv_linux/bin/activate")
    
    print("✅ Environment check completed")
    return True

def launch_jupyter():
    """Launch Jupyter notebook"""
    print_header("Launching Jupyter Notebook")
    
    if not check_environment():
        return
    
    print("🚀 Starting Jupyter Notebook server...")
    print("📓 Opening brain_tumor_analysis.ipynb")
    print("🌐 The notebook will open in your default web browser")
    print("\n💡 Tips:")
    print("   - Use Shift+Enter to run cells")
    print("   - Check GPU status in the first cell")
    print("   - Run training cells to see progress")
    print("   - Use Ctrl+C to stop the server")
    
    try:
        # Launch Jupyter notebook
        subprocess.run([
            "jupyter", "notebook", 
            "brain_tumor_analysis.ipynb",
            "--no-browser",  # We'll open it manually
            "--port=8888"
        ])
    except KeyboardInterrupt:
        print("\n👋 Jupyter server stopped")
    except Exception as e:
        print(f"❌ Error launching Jupyter: {e}")
        print("Please make sure Jupyter is installed: pip install jupyter")

def main():
    """Main function"""
    print_header("Brain Tumor Analysis - Jupyter Notebook")
    print("BRACU CSE428 Academic Project")
    print("Interactive Analysis Environment")
    
    print("\n🎯 This will launch Jupyter Notebook with the brain tumor analysis notebook")
    print("📊 You'll be able to:")
    print("   - Run training with real-time progress")
    print("   - Visualize results and plots")
    print("   - Experiment with different models")
    print("   - Analyze the dataset")
    
    response = input("\n🚀 Launch Jupyter Notebook? (y/n): ").strip().lower()
    
    if response in ['y', 'yes']:
        launch_jupyter()
    else:
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
