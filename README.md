# Hytale Style Detector

A machine learning model that detects if a texture matches Hytale's art style. Trained on ~7,700 textures and exports to ONNX for browser-based inference.

**Try it live:** https://hedgehogform.github.io/hytale-style-checker/

## Features

- **CNN-based classification** - Custom convolutional neural network optimized for pixel art textures
- **GPU acceleration** - Supports AMD GPUs via DirectML and NVIDIA GPUs via CUDA
- **Web deployment** - Exports to ONNX format for use with ONNX Runtime Web
- **Browser demo** - Drag-and-drop web interface for testing textures
- **One-click deploy** - Automatic GitHub Pages deployment via `gh` CLI

## Model Performance

- **Validation Accuracy**: ~92%
- **Training Data**: ~4,200 Hytale textures, ~3,500 non-Hytale (Minecraft) textures
- **Input Size**: 64x64 pixels (nearest-neighbor interpolation for pixel art)

> **Note:** Take results with a grain of salt - the model can produce false positives. It was only trained on Hytale and Minecraft textures, so it may not generalize well to other art styles.

## Contributing Training Data

Have Hytale-approved textures that could improve the model? Open an issue and upload them - I'd be happy to add them to the training dataset!

## Usage

### Train the model
```bash
python main.py
```

### Export to ONNX (without retraining)
```bash
python main.py export
```

### Deploy to GitHub Pages
```bash
python main.py deploy
```

## Requirements

- Python 3.10-3.12
- PyTorch 2.0+
- AMD GPU (DirectML) or NVIDIA GPU (CUDA) recommended

## Dataset Structure

```
dataset/
  hytale/     # Hytale-styled textures
  wrong/      # Non-Hytale textures
```

## Output Files

- `model/best_model.pth` - PyTorch model weights
- `onnx_model/model.onnx` - ONNX model for web deployment
- `onnx_model/metadata.json` - Model metadata (class names, input size)
- `web_example.html` - Browser demo page
- `docs/` - GitHub Pages deployment folder
