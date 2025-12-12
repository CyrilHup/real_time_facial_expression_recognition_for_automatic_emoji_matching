"""
Export PyTorch Models to ONNX Format
=====================================
Converts trained .pth models to optimized .onnx format for faster inference.
No retraining needed - just conversion with same accuracy.

Usage:
    python export_to_onnx.py
    
Output:
    - Creates .onnx files next to each .pth file
    - Example: emotion_model_best.pth → emotion_model_best.onnx
"""

import torch
import torch.onnx
import os
import numpy as np
from pathlib import Path
from model import load_model_smart, FaceEmotionCNN, FaceEmotionCNN_SE

def export_model_to_onnx(pth_path: str, device: torch.device = torch.device('cpu')):
    """
    Export a single PyTorch model to ONNX format.
    
    Args:
        pth_path: Path to .pth model file
        device: Device to load model on (CPU recommended for export)
    
    Returns:
        Path to exported .onnx file or None if failed
    """
    try:
        print(f"\n{'='*70}")
        print(f"Exporting: {os.path.basename(pth_path)}")
        print(f"{'='*70}")
        
        # Load model using smart loader
        model, model_info = load_model_smart(pth_path, device)
        model.eval()
        
        print(f"  Architecture: {model_info['architecture']}")
        print(f"  Classes: {model_info['num_classes']}")
        print(f"  Input channels: {model_info['in_channels']}")
        
        # Determine input size based on channels
        if model_info['in_channels'] == 1:
            # FER2013/FER+ - grayscale 48x48
            input_size = 48
            input_shape = (1, 1, input_size, input_size)
        else:
            # AffectNet - RGB 75x75
            input_size = 75
            input_shape = (1, 3, input_size, input_size)
        
        # Create dummy input for tracing
        dummy_input = torch.randn(input_shape, device=device)
        
        # Generate output path
        onnx_path = pth_path.replace('.pth', '.onnx').replace('.pt', '.onnx')
        
        print(f"  Input shape: {input_shape}")
        print(f"  Output path: {os.path.basename(onnx_path)}")
        print(f"\n  Exporting to ONNX...")
        
        # Export to ONNX
        torch.onnx.export(
            model,                          # Model to export
            dummy_input,                    # Dummy input
            onnx_path,                      # Output path
            export_params=True,             # Store trained parameters
            opset_version=14,               # ONNX version (14 = good compatibility)
            do_constant_folding=True,       # Optimize constant folding
            input_names=['input'],          # Input tensor name
            output_names=['output'],        # Output tensor name
            dynamic_axes={
                'input': {0: 'batch_size'},     # Variable batch size
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify export
        file_size = os.path.getsize(onnx_path) / (1024 * 1024)  # MB
        print(f"  ✓ Export successful!")
        print(f"  File size: {file_size:.2f} MB")
        
        # Test ONNX model (optional verification)
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"  ✓ ONNX model validation passed")
        except ImportError:
            print(f"  ⚠ Install 'onnx' package to validate exported models")
        except Exception as e:
            print(f"  ⚠ ONNX validation warning: {e}")
        
        return onnx_path
        
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_pth_models() -> list:
    """Find all .pth/.pt model files in current directory"""
    current_dir = Path(__file__).parent
    models = []
    
    for ext in ['*.pth', '*.pt']:
        models.extend(current_dir.glob(ext))
    
    return sorted([str(m) for m in models])


def main():
    """Export all PyTorch models to ONNX format"""
    print(f"\n{'='*70}")
    print("PyTorch to ONNX Model Exporter")
    print(f"{'='*70}")
    
    # Find all models
    pth_models = find_pth_models()
    
    if not pth_models:
        print("\n✗ No .pth or .pt model files found in current directory")
        return
    
    print(f"\nFound {len(pth_models)} model(s) to export:")
    for i, model in enumerate(pth_models, 1):
        print(f"  [{i}] {os.path.basename(model)}")
    
    # Ask user for confirmation
    print(f"\n{'='*70}")
    response = input("Export all models to ONNX? [Y/n]: ").strip().lower()
    
    if response and response not in ['y', 'yes', '']:
        print("Export cancelled.")
        return
    
    # Export each model
    device = torch.device('cpu')  # Use CPU for export (better compatibility)
    print(f"\nUsing device: {device}")
    
    exported = []
    failed = []
    
    for pth_path in pth_models:
        onnx_path = export_model_to_onnx(pth_path, device)
        
        if onnx_path:
            exported.append(onnx_path)
        else:
            failed.append(pth_path)
    
    # Summary
    print(f"\n{'='*70}")
    print("Export Summary")
    print(f"{'='*70}")
    print(f"  ✓ Successfully exported: {len(exported)}/{len(pth_models)}")
    
    if exported:
        print("\n  Exported files:")
        for path in exported:
            print(f"    • {os.path.basename(path)}")
    
    if failed:
        print(f"\n  ✗ Failed exports: {len(failed)}")
        for path in failed:
            print(f"    • {os.path.basename(path)}")
    
    print(f"\n{'='*70}")
    print("Next steps:")
    print("  1. Install ONNX Runtime: pip install onnxruntime-gpu")
    print("  2. Run app with ONNX models for 2-3x faster inference")
    print("  3. Same accuracy, optimized speed!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
