"""
Hytale Style Detector - Training Script
Trains a CNN model to detect if a texture matches Hytale's art style.
Exports the model to ONNX format for web deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from pathlib import Path
import json

# Configuration
DATASET_DIR = Path("dataset")
MODEL_OUTPUT_DIR = Path("model")
ONNX_OUTPUT_DIR = Path("onnx_model")
IMG_SIZE = 64  # Hytale textures are typically 16x16, 32x32, or 64x64
BATCH_SIZE = 128  # Larger batch = better GPU utilization
EPOCHS = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001


def get_device():
    """Get the best available device (DirectML for AMD, CUDA for NVIDIA, or CPU)."""
    # Try DirectML first (AMD GPUs on Windows)
    try:
        import torch_directml
        dml_device = torch_directml.device()
        print(f"Using DirectML (AMD GPU): {torch_directml.device_name(0)}")
        return dml_device
    except ImportError:
        pass

    # Try CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")

    # Fall back to CPU
    print("Using CPU (no GPU acceleration)")
    return torch.device("cpu")


DEVICE = get_device()


class HytaleStyleCNN(nn.Module):
    """CNN model optimized for texture classification."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Second conv block
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Third conv block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Fourth conv block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def create_datasets():
    """Load and prepare training and validation datasets."""

    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),  # Converts to [0, 1] and CHW format
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    # Load full dataset to get class names and split
    full_dataset = datasets.ImageFolder(str(DATASET_DIR), transform=train_transform)
    class_names = full_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Class mapping: 0 = {class_names[0]}, 1 = {class_names[1]}")
    print(f"Total images: {len(full_dataset)}")

    # Split into train and validation
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply validation transform to val_dataset
    # We need to create a new dataset with val_transform for validation
    val_dataset_proper = datasets.ImageFolder(str(DATASET_DIR), transform=val_transform)
    val_indices = val_dataset.indices
    val_dataset = Subset(val_dataset_proper, val_indices)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    # Note: pin_memory=False for DirectML, num_workers=0 for Windows compatibility
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Windows + DirectML works best with 0
        pin_memory=False,
        drop_last=True,  # Drop incomplete batches for consistent GPU load
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, class_names


def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)

            outputs = model(images)
            # Compute loss on CPU to avoid DirectML fallback warning
            loss = criterion(outputs.cpu(), labels.cpu())

            running_loss += loss.item() * images.size(0)
            predicted = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # Calculate precision, recall, F1
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum().float()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().float()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().float()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return epoch_loss, epoch_acc, precision.item(), recall.item(), f1.item()


def train_model(model, train_loader, val_loader):
    """Train the model with early stopping."""

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    MODEL_OUTPUT_DIR.mkdir(exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Move to CPU before saving to avoid DirectML serialization issues
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, MODEL_OUTPUT_DIR / "best_model.pth")
            best_state_dict = cpu_state_dict  # Keep a copy
            print(f"  -> Saved best model (val_acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    # Load best model
    model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "best_model.pth", weights_only=True))
    return model


def export_to_onnx(model, class_names):
    """Export the trained model to ONNX format for web deployment."""

    ONNX_OUTPUT_DIR.mkdir(exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    onnx_path = ONNX_OUTPUT_DIR / "model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Exported ONNX model to {onnx_path}")

    # Save model metadata
    metadata = {
        "input_size": IMG_SIZE,
        "class_names": class_names,
        "class_mapping": {
            "0": class_names[0],
            "1": class_names[1]
        },
        "threshold": 0.5
    }

    with open(ONNX_OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {ONNX_OUTPUT_DIR / 'metadata.json'}")


def create_web_example():
    """Create an example HTML/JS file showing how to use the ONNX model on the web."""

    example_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hytale Style Texture Checker</title>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { text-align: center; color: #00d4aa; }
        .drop-zone {
            border: 3px dashed #00d4aa;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin: 20px 0;
        }
        .drop-zone:hover, .drop-zone.dragover {
            background: rgba(0, 212, 170, 0.1);
            border-color: #00ffcc;
        }
        .preview {
            max-width: 200px;
            margin: 20px auto;
            display: block;
            image-rendering: pixelated;
            border: 2px solid #333;
        }
        .result {
            text-align: center;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            font-size: 1.2em;
        }
        .result.hytale {
            background: linear-gradient(135deg, #00d4aa22, #00d4aa44);
            border: 2px solid #00d4aa;
        }
        .result.wrong {
            background: linear-gradient(135deg, #ff6b6b22, #ff6b6b44);
            border: 2px solid #ff6b6b;
        }
        .confidence { font-size: 0.9em; opacity: 0.8; margin-top: 10px; }
        .loading { opacity: 0.5; }
        #status { text-align: center; margin: 10px 0; color: #888; }
    </style>
</head>
<body>
    <h1>Hytale Style Checker</h1>
    <p id="status">Loading model...</p>

    <div class="drop-zone" id="dropZone">
        <p>Drop a texture image here<br>or click to select</p>
        <input type="file" id="fileInput" accept="image/*" hidden>
    </div>

    <img id="preview" class="preview" hidden>
    <div id="result" class="result" hidden></div>

    <script>
        const IMG_SIZE = 64;
        let session = null;
        let metadata = null;

        // Load the ONNX model
        async function loadModel() {
            try {
                // Load metadata
                const metaResponse = await fetch('./onnx_model/metadata.json');
                metadata = await metaResponse.json();

                // Load ONNX model
                session = await ort.InferenceSession.create('./onnx_model/model.onnx');
                document.getElementById('status').textContent = 'Model loaded! Drop a texture to check.';
            } catch (error) {
                document.getElementById('status').textContent = 'Error loading model: ' + error.message;
                console.error(error);
            }
        }

        // Preprocess image for the model
        function preprocessImage(img) {
            // Create canvas to resize image
            const canvas = document.createElement('canvas');
            canvas.width = IMG_SIZE;
            canvas.height = IMG_SIZE;
            const ctx = canvas.getContext('2d');

            // Disable image smoothing for pixel art
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

            // Get image data
            const imageData = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
            const data = imageData.data;

            // Convert to CHW format and normalize to [0, 1]
            const float32Data = new Float32Array(3 * IMG_SIZE * IMG_SIZE);

            for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
                float32Data[i] = data[i * 4] / 255.0;                    // R
                float32Data[IMG_SIZE * IMG_SIZE + i] = data[i * 4 + 1] / 255.0;  // G
                float32Data[2 * IMG_SIZE * IMG_SIZE + i] = data[i * 4 + 2] / 255.0;  // B
            }

            return new ort.Tensor('float32', float32Data, [1, 3, IMG_SIZE, IMG_SIZE]);
        }

        // Predict if texture is Hytale style
        async function predict(img) {
            if (!session) {
                alert('Model not loaded yet!');
                return;
            }

            const tensor = preprocessImage(img);
            const results = await session.run({ input: tensor });
            const score = results.output.data[0];

            // Score >= 0.5 means class 1 (check metadata for class names)
            // In ImageFolder, classes are sorted alphabetically: hytale=0, wrong=1
            // So score >= 0.5 means "wrong", score < 0.5 means "hytale"
            const isHytale = score < 0.5;

            return {
                isHytale: isHytale,
                confidence: isHytale ? (1 - score) : score,
                rawScore: score
            };
        }

        // Handle file selection
        async function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please select an image file');
                return;
            }

            const img = new Image();
            const reader = new FileReader();

            reader.onload = async (e) => {
                img.onload = async () => {
                    // Show preview
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.hidden = false;

                    // Get prediction
                    const result = await predict(img);

                    // Show result
                    const resultDiv = document.getElementById('result');
                    resultDiv.hidden = false;
                    resultDiv.className = 'result ' + (result.isHytale ? 'hytale' : 'wrong');
                    resultDiv.innerHTML = `
                        <strong>${result.isHytale ? 'Hytale Style!' : 'Not Hytale Style'}</strong>
                        <div class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
                    `;
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }

        // Set up drag and drop
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');

        dropZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });

        // Initialize
        loadModel();
    </script>
</body>
</html>'''

    with open("web_example.html", "w") as f:
        f.write(example_html)
    print("Created web_example.html")


def main():
    """Main training pipeline."""

    print("=" * 50)
    print("Hytale Style Detector - Training (PyTorch)")
    print("=" * 50)

    print(f"\nDevice: {DEVICE}")

    print(f"\nConfiguration:")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Validation split: {VALIDATION_SPLIT * 100}%")
    print(f"  Learning rate: {LEARNING_RATE}")

    # Load datasets
    print("\nLoading datasets...")
    train_loader, val_loader, class_names = create_datasets()

    # Create model
    print("\nCreating model...")
    model = HytaleStyleCNN().to(DEVICE)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train model
    print("\nStarting training...")
    model = train_model(model, train_loader, val_loader)

    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)
    criterion = nn.BCELoss()
    val_loss, val_acc, precision, recall, f1 = validate(model, val_loader, criterion)
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.1f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Export to ONNX
    print("\nExporting model for web deployment...")
    export_to_onnx(model, class_names)

    # Save PyTorch model
    torch.save(model.state_dict(), MODEL_OUTPUT_DIR / "final_model.pth")
    print(f"Saved PyTorch model to {MODEL_OUTPUT_DIR / 'final_model.pth'}")

    # Create web example
    create_web_example()

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    print(f"\nFiles created:")
    print(f"  - {MODEL_OUTPUT_DIR}/best_model.pth (best checkpoint)")
    print(f"  - {MODEL_OUTPUT_DIR}/final_model.pth (final model)")
    print(f"  - {ONNX_OUTPUT_DIR}/model.onnx (ONNX model for web)")
    print(f"  - {ONNX_OUTPUT_DIR}/metadata.json (model metadata)")
    print(f"  - web_example.html (example web page)")
    print(f"\nTo use on your website:")
    print(f"  1. Copy the '{ONNX_OUTPUT_DIR}' folder to your web server")
    print(f"  2. Use the code in 'web_example.html' as a reference")
    print(f"  3. Load the model with ONNX Runtime Web")


def export_only():
    """Load existing model and export to ONNX without retraining."""
    print("=" * 50)
    print("Exporting existing model to ONNX")
    print("=" * 50)

    # Load model on CPU for export
    model = HytaleStyleCNN()
    model.load_state_dict(torch.load(MODEL_OUTPUT_DIR / "best_model.pth", weights_only=True))
    model.to("cpu")

    # Get class names from dataset folder
    class_names = sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
    print(f"Classes: {class_names}")

    # Export
    export_to_onnx_cpu(model, class_names)
    create_web_example()

    print("\nExport complete!")
    print(f"Files created:")
    print(f"  - {ONNX_OUTPUT_DIR}/model.onnx")
    print(f"  - {ONNX_OUTPUT_DIR}/metadata.json")
    print(f"  - web_example.html")


def export_to_onnx_cpu(model, class_names):
    """Export model to ONNX on CPU."""
    ONNX_OUTPUT_DIR.mkdir(exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    onnx_path = ONNX_OUTPUT_DIR / "model.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Exported ONNX model to {onnx_path}")

    # Save model metadata
    metadata = {
        "input_size": IMG_SIZE,
        "class_names": class_names,
        "class_mapping": {
            "0": class_names[0],
            "1": class_names[1]
        },
        "threshold": 0.5
    }

    with open(ONNX_OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {ONNX_OUTPUT_DIR / 'metadata.json'}")


def deploy_github_pages():
    """Deploy the model and web page to GitHub Pages."""
    import shutil

    PAGES_DIR = Path("docs")

    print("=" * 50)
    print("Deploying to GitHub Pages")
    print("=" * 50)

    # Check if model exists
    if not (ONNX_OUTPUT_DIR / "model.onnx").exists():
        print("Error: ONNX model not found. Run 'python main.py export' first.")
        return

    # Create docs directory (GitHub Pages serves from /docs)
    if PAGES_DIR.exists():
        shutil.rmtree(PAGES_DIR)
    PAGES_DIR.mkdir()

    # Copy ONNX model and metadata
    onnx_dest = PAGES_DIR / "onnx_model"
    onnx_dest.mkdir()
    shutil.copy(ONNX_OUTPUT_DIR / "model.onnx", onnx_dest / "model.onnx")
    shutil.copy(ONNX_OUTPUT_DIR / "metadata.json", onnx_dest / "metadata.json")
    print(f"Copied model files to {onnx_dest}")

    # Copy and rename HTML to index.html
    if Path("web_example.html").exists():
        shutil.copy("web_example.html", PAGES_DIR / "index.html")
        print(f"Copied web_example.html to {PAGES_DIR / 'index.html'}")
    else:
        print("Error: web_example.html not found. Run 'python main.py export' first.")
        return

    # Create .nojekyll file to prevent Jekyll processing
    (PAGES_DIR / ".nojekyll").touch()

    print(f"\nGitHub Pages files created in '{PAGES_DIR}/' directory!")
    print("\nNext steps:")
    print("  1. Commit and push the 'docs' folder to your repository")
    print("  2. Go to your repo Settings > Pages")
    print("  3. Set Source to 'Deploy from a branch'")
    print("  4. Select 'master' (or main) branch and '/docs' folder")
    print("  5. Save and wait for deployment")
    print("\nYour site will be available at:")
    print("  https://<username>.github.io/<repo-name>/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "export":
            export_only()
        elif sys.argv[1] == "deploy":
            deploy_github_pages()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage:")
            print("  python main.py         - Train the model")
            print("  python main.py export  - Export existing model to ONNX")
            print("  python main.py deploy  - Deploy to GitHub Pages")
    else:
        main()
