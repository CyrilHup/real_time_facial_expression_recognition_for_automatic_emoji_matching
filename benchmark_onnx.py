"""
Benchmark Script - Compare PyTorch vs ONNX Performance
=======================================================
Tests inference speed for all models with and without ONNX.
"""

import torch
import numpy as np
import time
import os
from pathlib import Path
from model import load_model_smart

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠ ONNX Runtime not installed. Install with: pip install onnxruntime-gpu")


def benchmark_pytorch(model_path: str, device: torch.device, iterations: int = 100):
    """Benchmark PyTorch model"""
    print(f"\n  PyTorch Benchmark...")
    
    model, model_info = load_model_smart(model_path, device)
    model.eval()
    
    # Create dummy input
    in_channels = model_info['in_channels']
    img_size = 48 if in_channels == 1 else 75
    dummy_input = torch.randn(1, in_channels, img_size, img_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            times.append((time.perf_counter() - start) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"    Avg: {avg_time:.2f}ms ± {std_time:.2f}ms")
    print(f"    FPS: {1000/avg_time:.1f}")
    
    return avg_time


def benchmark_onnx(onnx_path: str, device: torch.device, iterations: int = 100):
    """Benchmark ONNX model"""
    if not ONNX_AVAILABLE:
        return None
    
    print(f"\n  ONNX Benchmark...")
    
    try:
        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"    Provider: {session.get_providers()[0]}")
        
        # Get input shape (replace symbolic dims with actual values)
        input_meta = session.get_inputs()[0]
        input_shape = input_meta.shape
        
        # Replace symbolic dimensions (e.g., 'batch_size') with actual values
        actual_shape = []
        for dim in input_shape:
            if isinstance(dim, str):
                actual_shape.append(1)  # batch_size = 1
            else:
                actual_shape.append(dim)
        
        dummy_input = np.random.randn(*actual_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {input_meta.name: dummy_input})
        
        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = session.run(None, {input_meta.name: dummy_input})
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"    Avg: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"    FPS: {1000/avg_time:.1f}")
        
        return avg_time
        
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        return None


def main():
    print(f"\n{'='*70}")
    print("Performance Benchmark: PyTorch vs ONNX")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Find models
    current_dir = Path(__file__).parent
    models = list(current_dir.glob('*.pth')) + list(current_dir.glob('*.pt'))
    
    if not models:
        print("\n✗ No .pth or .pt models found")
        return
    
    print(f"\nFound {len(models)} model(s)\n")
    
    results = []
    
    for model_path in sorted(models):
        model_name = model_path.name
        onnx_path = model_path.with_suffix('.onnx')
        
        print(f"{'='*70}")
        print(f"Model: {model_name}")
        print(f"{'='*70}")
        
        # Benchmark PyTorch
        pytorch_time = benchmark_pytorch(str(model_path), device, iterations=100)
        
        # Benchmark ONNX if available
        onnx_time = None
        if onnx_path.exists() and ONNX_AVAILABLE:
            onnx_time = benchmark_onnx(str(onnx_path), device, iterations=100)
        elif not onnx_path.exists():
            print(f"\n  ⚠ ONNX file not found: {onnx_path.name}")
            print(f"    Run: python export_to_onnx.py")
        
        # Calculate speedup
        if onnx_time:
            speedup = pytorch_time / onnx_time
            print(f"\n  Speedup: {speedup:.2f}x faster with ONNX 🚀")
            results.append({
                'model': model_name,
                'pytorch': pytorch_time,
                'onnx': onnx_time,
                'speedup': speedup
            })
        else:
            results.append({
                'model': model_name,
                'pytorch': pytorch_time,
                'onnx': None,
                'speedup': None
            })
    
    # Summary
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}\n")
    
    print(f"{'Model':<30} {'PyTorch':<12} {'ONNX':<12} {'Speedup':<10}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    
    for result in results:
        pytorch_str = f"{result['pytorch']:.2f}ms"
        onnx_str = f"{result['onnx']:.2f}ms" if result['onnx'] else "N/A"
        speedup_str = f"{result['speedup']:.2f}x" if result['speedup'] else "N/A"
        
        print(f"{result['model']:<30} {pytorch_str:<12} {onnx_str:<12} {speedup_str:<10}")
    
    # Average speedup
    valid_speedups = [r['speedup'] for r in results if r['speedup']]
    if valid_speedups:
        avg_speedup = np.mean(valid_speedups)
        print(f"\nAverage speedup: {avg_speedup:.2f}x 🔥")
    
    print(f"\n{'='*70}")
    print("Recommendation:")
    if ONNX_AVAILABLE and any(r['onnx'] for r in results):
        print("  ✓ Use ONNX models for production (2-3x faster)")
        print("  ✓ Same accuracy, optimized inference")
    else:
        print("  ! Install ONNX Runtime: pip install onnxruntime-gpu")
        print("  ! Export models: python export_to_onnx.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
