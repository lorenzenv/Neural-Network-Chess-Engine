#!/usr/bin/env python3
"""
Deployment script to automatically use the latest engine version
"""

import os
import shutil
import glob

def find_latest_engine():
    """Find the latest engine version file"""
    engine_files = glob.glob("chess_engine_v*.py")
    
    if not engine_files:
        print("No versioned engine files found!")
        return None
    
    # Extract version numbers and find the highest
    versions = []
    for filename in engine_files:
        try:
            version_str = filename.replace("chess_engine_v", "").replace(".py", "")
            version_num = float(version_str)
            versions.append((version_num, filename))
        except ValueError:
            continue
    
    if not versions:
        print("No valid version numbers found!")
        return None
    
    # Sort by version number and get the latest
    versions.sort(reverse=True)
    latest_version, latest_file = versions[0]
    
    return latest_file, latest_version

def deploy_latest():
    """Deploy the latest engine version"""
    latest_info = find_latest_engine()
    
    if not latest_info:
        print("❌ No latest engine found to deploy!")
        return False
    
    latest_file, latest_version = latest_info
    
    print(f"🚀 Found latest engine: {latest_file} (v{latest_version})")
    
    # Create backup of current production engine
    if os.path.exists("chess_engine.py"):
        backup_name = "chess_engine_backup_production.py"
        shutil.copy2("chess_engine.py", backup_name)
        print(f"📦 Backed up current engine to {backup_name}")
    
    # Copy latest version to production
    shutil.copy2(latest_file, "chess_engine.py")
    
    # Update the import in app.py to use the latest version
    update_app_import(latest_file, latest_version)
    
    print(f"✅ Successfully deployed {latest_file} as production engine!")
    print(f"🎯 Production engine is now v{latest_version}")
    
    return True

def update_app_import(latest_file, version):
    """Update app.py to import from the latest engine"""
    with open("app.py", "r") as f:
        content = f.read()
    
    # Replace the import line
    engine_module = latest_file.replace(".py", "")
    class_name = "EngineV" + str(int(version)) if version >= 2.0 else "Engine"
    
    # Update imports
    import_line = f"from {engine_module} import {class_name}, ENGINE_VERSION, ENGINE_NAME, ENGINE_FEATURES"
    
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith("from chess_engine"):
            lines[i] = import_line
            break
    
    # Update engine instantiation
    for i, line in enumerate(lines):
        if "= Engine" in line and "fen)" in line:
            lines[i] = f"    engine = {class_name}(fen)"
            break
    
    # Write back
    with open("app.py", "w") as f:
        f.write('\n'.join(lines))
    
    print(f"📝 Updated app.py to use {class_name} from {engine_module}")

if __name__ == "__main__":
    print("🚀 NEURAL CHESS ENGINE DEPLOYMENT")
    print("=" * 40)
    
    success = deploy_latest()
    
    if success:
        print("\n🎉 Deployment complete!")
        print("💡 Your app now uses the latest engine version!")
        print("🔄 Restart your Flask app to see the changes.")
    else:
        print("\n❌ Deployment failed!") 