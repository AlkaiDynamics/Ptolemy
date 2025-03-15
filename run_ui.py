"""
PTOLEMY UI Launcher

This script launches the PTOLEMY web interface using FastAPI and Uvicorn.
"""
import uvicorn
import os
import sys
from pathlib import Path

# Add the project root to the path if running as a script
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("Starting PTOLEMY UI...")
    print("Access the interface at http://127.0.0.1:8000")
    uvicorn.run("ptolemy.ui.app:app", host="127.0.0.1", port=8000, reload=True)
