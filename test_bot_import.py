#!/usr/bin/env python3
"""
Test script to verify Lichess bot imports work correctly
"""

import sys
import os

def test_imports():
    """Test all required imports for the Lichess bot"""
    try:
        print("🔍 Testing imports...")
        
        # Test basic imports
        import chess
        print("✅ chess library imported")
        
        import berserk
        print("✅ berserk library imported")
        
        # Test TensorFlow/TFLite import (handle both environments)
        try:
            import tflite_runtime.interpreter as tflite
            print("✅ tflite_runtime imported (production)")
        except ImportError:
            try:
                import tensorflow as tf
                print("✅ tensorflow imported (development)")
            except ImportError:
                print("❌ Neither tflite_runtime nor tensorflow available")
                return False
        
        # Test engine import
        from chess_engine_v4 import EngineV4, ENGINE_VERSION, ENGINE_NAME, ENGINE_FEATURES
        print(f"✅ V4 engine imported: {ENGINE_NAME} v{ENGINE_VERSION}")
        
        # Test environment variable
        token = os.getenv('LICHESS_TOKEN') or os.getenv('LICHESS_API_TOKEN')
        if token:
            print(f"✅ API token found: {token[:8]}...")
        else:
            print("⚠️  No API token found (set LICHESS_API_TOKEN)")
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 