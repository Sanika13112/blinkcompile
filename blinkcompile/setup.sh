#!/bin/bash

# Optimized setup script for Streamlit Cloud
# This script installs dependencies with minimal memory usage

echo "🚀 Starting optimized installation for Streamlit Cloud..."

# Upgrade pip
pip install --upgrade pip

# Install core dependencies first
echo "📦 Installing core dependencies..."
pip install streamlit==1.28.0
pip install Pillow==10.0.0
pip install qrcode==7.4.2
pip install psutil==5.9.5
pip install numpy==1.23.5

# Install TensorFlow CPU version (lighter than full TensorFlow)
echo "🤖 Installing TensorFlow CPU..."
pip install tensorflow-cpu==2.10.0

# Install Hugging Face transformers without unnecessary extras
echo "🧠 Installing Hugging Face transformers..."
pip install transformers==4.30.0

# Install ONNX runtime
echo "🔄 Installing ONNX..."
pip install onnx==1.13.1

# Install PyTorch CPU version (lighter)
echo "🔥 Installing PyTorch CPU..."
pip install torch==1.13.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Install additional required packages
echo "📊 Installing additional packages..."
pip install safetensors==0.3.1
pip install accelerate==0.18.0

echo "✅ Installation complete!"
echo "📦 Installed packages:"
pip list