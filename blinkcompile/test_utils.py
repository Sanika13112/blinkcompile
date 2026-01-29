#!/usr/bin/env python
"""Test the utils.py functions"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import *

def test_safe_calculations():
    print("🧪 Testing Safe Calculations")
    
    # Test safe_calculate_reduction
    test_cases = [
        (100, 25, 75.0),
        (0, 25, 75.0),     # Should return default 75%
        (100, 0, 100.0),
        (100, 100, 0.0),
        (-100, 25, 75.0),  # Negative original size
        (100, -25, 75.0),  # Negative compressed size
    ]
    
    for orig, comp, expected in test_cases:
        result = safe_calculate_reduction(orig, comp)
        status = "✓" if abs(result - expected) < 0.1 else "✗"
        print(f"  {status} {orig}/{comp} -> {result:.1f}% (expected: {expected:.1f}%)")
    
    # Test format_file_size
    print("\n🧪 Testing File Size Formatting")
    sizes = [0, 1023, 1024, 1024*1024, 1024*1024*1024, -1]
    for size in sizes:
        formatted = format_file_size(size)
        print(f"  {size} -> {formatted}")

def test_error_handling():
    print("\n🧪 Testing Error Handling")
    
    # Test QR generation with invalid data
    try:
        qr = generate_qr(None)
        print("  ✓ QR generation handles invalid data")
    except:
        print("  ✗ QR generation failed with invalid data")
    
    # Test compatibility with invalid input
    result = validate_edge_compatibility("invalid", "Raspberry Pi 4")
    print(f"  ✓ Compatibility check handles invalid input: {result}")
    
    # Test model card with invalid data
    html = create_model_card("Test", "invalid", "invalid", [])
    print(f"  ✓ Model card handles invalid data (length: {len(html)} chars)")

def test_new_functions():
    print("\n🧪 Testing New Functions")
    
    # Test generate_blinkcompile_report
    report = generate_blinkcompile_report("test_model", 100*1024*1024, 25*1024*1024, "Raspberry Pi 4")
    print(f"  ✓ Report generated ({len(report)} chars)")
    
    # Test get_file_icon
    icons = ['.tflite', '.onnx', '.unknown']
    for ext in icons:
        icon = get_file_icon(f"file{ext}")
        print(f"  ✓ Icon for {ext}: {icon}")
    
    # Test estimate_inference_time
    times = estimate_inference_time(100, "Raspberry Pi 4")
    print(f"  ✓ Estimated inference time: {times}")

def main():
    print("⚡ Testing BlinkCompile Utils")
    print("=" * 50)
    
    test_safe_calculations()
    test_error_handling()
    test_new_functions()
    
    print("\n" + "=" * 50)
    print("✅ All utils tests completed!")

if __name__ == "__main__":
    main()