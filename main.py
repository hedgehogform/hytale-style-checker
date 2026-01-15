"""
Hytale Style Detector - Training Script
Trains a CNN model to detect if a texture matches Hytale's art style.
Exports the model to ONNX format for web deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from pathlib import Path
from PIL import Image
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

# Explicitly define the correct style folder
# All other folders in the dataset will be treated as "wrong" style
CORRECT_STYLE_FOLDER = "hytale"


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


class BinaryStyleDataset(Dataset):
    """Dataset that treats one folder as 'correct' (label 0) and all others as 'wrong' (label 1)."""

    def __init__(self, root_dir, correct_folder, transform=None):
        self.transform = transform
        self.samples = []  # List of (path, label)
        self.bad_files = []  # Track files that couldn't be loaded

        root = Path(root_dir)
        folders = [f for f in root.iterdir() if f.is_dir()]

        for folder in folders:
            # Label 0 = correct style (hytale), Label 1 = wrong style (everything else)
            label = 0 if folder.name == correct_folder else 1

            # Find all images in this folder recursively
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.webp']:
                for img_path in folder.rglob(ext):
                    if self._is_valid_image(img_path):
                        self.samples.append((img_path, label))
                for img_path in folder.rglob(ext.upper()):
                    if self._is_valid_image(img_path):
                        self.samples.append((img_path, label))

        # Count samples per class
        self.correct_count = sum(1 for _, l in self.samples if l == 0)
        self.wrong_count = sum(1 for _, l in self.samples if l == 1)

        # Report bad files
        if self.bad_files:
            print(f"Warning: Skipped {len(self.bad_files)} invalid/corrupted images:")
            for f in self.bad_files[:5]:  # Show first 5
                print(f"  - {f}")
            if len(self.bad_files) > 5:
                print(f"  ... and {len(self.bad_files) - 5} more")

    def _is_valid_image(self, img_path):
        """Check if file is a valid image that can be opened."""
        try:
            with Image.open(img_path) as img:
                img.verify()
            return True
        except Exception:
            self.bad_files.append(img_path)
            return False

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def create_datasets():
    """Load and prepare training and validation datasets."""

    # Training transforms with augmentation
    # Use BILINEAR for general images - works for both textures and screenshots
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),  # Converts to [0, 1] and CHW format
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    # Load full dataset with explicit label mapping
    full_dataset = BinaryStyleDataset(DATASET_DIR, CORRECT_STYLE_FOLDER, transform=train_transform)

    # List all folders for display
    folders = [f.name for f in DATASET_DIR.iterdir() if f.is_dir()]
    print(f"Folders found: {folders}")
    print(f"Correct style folder: '{CORRECT_STYLE_FOLDER}' -> Label 0")
    print(f"Wrong style folders: {[f for f in folders if f != CORRECT_STYLE_FOLDER]} -> Label 1")
    print(f"Correct style images: {full_dataset.correct_count}")
    print(f"Wrong style images: {full_dataset.wrong_count}")
    print(f"Total images: {len(full_dataset)}")

    # Split into train and validation
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create validation dataset with proper transform
    val_dataset_proper = BinaryStyleDataset(DATASET_DIR, CORRECT_STYLE_FOLDER, transform=val_transform)
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

    # Class names: 0 = correct (hytale), 1 = wrong
    class_names = [CORRECT_STYLE_FOLDER, "wrong"]

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
    model = HytaleStyleCNN()

    # Load existing model if available (continue training)
    existing_model_path = MODEL_OUTPUT_DIR / "best_model.pth"
    if existing_model_path.exists():
        print(f"Loading existing model from {existing_model_path} (continuing training)...")
        model.load_state_dict(torch.load(existing_model_path, weights_only=True))

    model = model.to(DEVICE)

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

    # Class names: 0 = correct (hytale), 1 = wrong
    class_names = [CORRECT_STYLE_FOLDER, "wrong"]
    print(f"Class mapping: 0 = '{CORRECT_STYLE_FOLDER}' (correct), 1 = 'wrong'")

    # Export
    export_to_onnx_cpu(model, class_names)

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
    """Deploy the model and web page to GitHub Pages using gh CLI."""
    import shutil
    import subprocess

    PAGES_DIR = Path("docs")

    print("=" * 50)
    print("Deploying to GitHub Pages")
    print("=" * 50)

    # Check if gh CLI is available
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: GitHub CLI (gh) not found. Install it from https://cli.github.com/")
        return

    # Check if model exists
    if not (ONNX_OUTPUT_DIR / "model.onnx").exists():
        print("Error: ONNX model not found. Run 'python main.py export' first.")
        return

    if not Path("web_example.html").exists():
        print("Error: web_example.html not found. Run 'python main.py export' first.")
        return

    # Create docs directory (GitHub Pages serves from /docs)
    print("\nPreparing files...")
    if PAGES_DIR.exists():
        shutil.rmtree(PAGES_DIR)
    PAGES_DIR.mkdir()

    # Copy ONNX model and metadata
    onnx_dest = PAGES_DIR / "onnx_model"
    onnx_dest.mkdir()
    shutil.copy(ONNX_OUTPUT_DIR / "model.onnx", onnx_dest / "model.onnx")
    shutil.copy(ONNX_OUTPUT_DIR / "metadata.json", onnx_dest / "metadata.json")
    print(f"  Copied model files to {onnx_dest}")

    # Copy and rename HTML to index.html
    shutil.copy("web_example.html", PAGES_DIR / "index.html")
    print(f"  Copied web_example.html to {PAGES_DIR / 'index.html'}")

    # Create .nojekyll file to prevent Jekyll processing
    (PAGES_DIR / ".nojekyll").touch()

    # Git operations
    print("\nCommitting changes...")
    subprocess.run(["git", "add", "docs/"], check=True)
    subprocess.run(["git", "commit", "-m", "Deploy to GitHub Pages"], check=False)  # May fail if no changes

    # Get current branch
    result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
    branch = result.stdout.strip() or "master"

    # Push to remote
    print(f"\nPushing to remote ({branch})...")
    subprocess.run(["git", "push", "origin", branch], check=True)

    # Get repo info
    result = subprocess.run(["gh", "repo", "view", "--json", "name,owner"], capture_output=True, text=True)
    if result.returncode == 0:
        import json as json_module
        repo_info = json_module.loads(result.stdout)
        repo_name = repo_info["name"]
        owner = repo_info["owner"]["login"]
    else:
        print("Warning: Could not get repo info")
        repo_name = "<repo>"
        owner = "<username>"

    # Enable GitHub Pages via gh CLI
    print("\nEnabling GitHub Pages...")
    result = subprocess.run(
        ["gh", "api", "-X", "POST", f"/repos/{owner}/{repo_name}/pages",
         "-f", f"source[branch]={branch}", "-f", "source[path]=/docs"],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        # Pages might already be enabled, try updating instead
        subprocess.run(
            ["gh", "api", "-X", "PUT", f"/repos/{owner}/{repo_name}/pages",
             "-f", f"source[branch]={branch}", "-f", "source[path]=/docs"],
            capture_output=True, text=True
        )

    print("\n" + "=" * 50)
    print("Deployment complete!")
    print("=" * 50)
    print("\nYour site will be available at:")
    print(f"  https://{owner}.github.io/{repo_name}/")
    print("\nNote: It may take a few minutes for the site to be live.")


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
