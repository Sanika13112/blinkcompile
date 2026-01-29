"""
Configuration settings for EdgeFlow Compressor
"""

# Edge device profiles
DEVICE_PROFILES = {
    "Raspberry Pi 4": {
        "ram": "4-8GB",
        "storage": "16-32GB",
        "cpu": "ARM Cortex-A72",
        "compatible_formats": ["TFLite", "ONNX"],
        "max_model_size_mb": 100
    },
    "Android Phone": {
        "ram": "4-12GB",
        "storage": "64-256GB",
        "cpu": "ARM Cortex",
        "compatible_formats": ["TFLite", "TensorFlow.js"],
        "max_model_size_mb": 50
    },
    "iPhone": {
        "ram": "4-8GB",
        "storage": "64-512GB",
        "cpu": "Apple Silicon",
        "compatible_formats": ["CoreML", "ONNX"],
        "max_model_size_mb": 50
    },
    "Jetson Nano": {
        "ram": "4GB",
        "storage": "16GB",
        "cpu": "ARM Cortex-A57",
        "compatible_formats": ["TFLite", "ONNX", "TensorRT"],
        "max_model_size_mb": 200
    }
}

# Popular edge-optimized models
POPULAR_MODELS = {
    "Vision": [
        "google/mobilenet_v2_1.0_224",
        "apple/mobilevit-small",
        "google/vit-base-patch16-224",
        "microsoft/resnet-50"
    ],
    "NLP": [
        "distilbert-base-uncased",
        "google/mobilebert-uncased",
        "microsoft/xtremedistil-l6-h256-uncased"
    ],
    "Audio": [
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        "facebook/wav2vec2-base-960h"
    ]
}

# Compression presets
COMPRESSION_PRESETS = {
    "Max Performance": {
        "quantization": "int8",
        "pruning": 0.5,
        "optimization_level": 3
    },
    "Balanced": {
        "quantization": "float16",
        "pruning": 0.3,
        "optimization_level": 2
    },
    "Minimal Size": {
        "quantization": "int8",
        "pruning": 0.7,
        "optimization_level": 3
    },
    "Max Accuracy": {
        "quantization": None,
        "pruning": 0.0,
        "optimization_level": 1
    }
}