try:
    import torch
    print(f"✅ PyTorch is present! Version: {torch.__version__}")
    
    # Optional: Check if GPU/CUDA is available
    if torch.cuda.is_available():
        print(f"🔥 CUDA (GPU) is available: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Running on CPU only (CUDA not available).")
        
except ImportError:
    print("❌ PyTorch is NOT installed in this environment.")
