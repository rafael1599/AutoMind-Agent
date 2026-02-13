import os
import sys
import multiprocessing

def check_system():
    print("="*40)
    print("     HARDWARE & SOFTWARE CHECK")
    print("="*40)
    
    # 1. CPU Check (i9-10900KF Target)
    # Note: multiprocessing.cpu_count() returns logical cores
    logical_cores = multiprocessing.cpu_count()
    
    # Estimate physical cores (assuming hyperthreading with 2 threads per core)
    physical_cores = logical_cores // 2
    
    print(f"[CPU] Estimated Physical Cores: {physical_cores}")
    print(f"[CPU] Logical Threads: {logical_cores}")
    
    # Validation for i9-10900KF (10 cores / 20 threads)
    if physical_cores >= 10 and logical_cores >= 20:
        print("✅ HARDWARE VERIFIED: High-performance CPU detected (likely i9-10900KF or better).")
    elif logical_cores >= 16:
        print("⚠️ HARDWARE NOTE: High thread count detected. Suitable for parallel RL training.")
    else:
        print("⚠️ HARDWARE WARNING: Core count lower than expected for i9-10900KF.")

    print("-"*40)

    # 2. Software Dependencies
    dependencies = {
        "torch": "PyTorch",
        "stable_baselines3": "Stable Baselines 3",
        "gymnasium": "Gymnasium",
        "numpy": "NumPy"
    }

    all_installed = True
    for package, name in dependencies.items():
        try:
            __import__(package)
            version = sys.modules[package].__version__
            print(f"✅ FOUND: {name} (v{version})")
        except ImportError:
            print(f"❌ MISSING: {name}")
            all_installed = False

    print("-"*40)
    
    # 3. CUDA Check (Optional for RL on CPU, but good to know)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA AVAILABLE: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA NOT AVAILABLE: Training will rely on CPU (Optimized for i9).")
    except ImportError:
        pass

    print("="*40)
    
    if all_installed:
        print("🚀 READY FOR TRAINING")
    else:
        print("🛑 ACTION REQUIRED: Install missing dependencies.")

if __name__ == "__main__":
    check_system()
