"""
SpeciesNet Fine-Tuning for 15 Species (Fixed Input Shape)
"""
import os
import time, math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from speciesnet.classifier import SpeciesNetClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


# Configuration
MODEL_NAME = "kaggle:google/speciesnet/pyTorch/v4.0.1a"
MODEL_WEIGHTS = "/data/code/SpeciesNet/random/speciesnet_15_2init.pth"
TRAIN_DATA = "/data/AWT/random_datasets_from_snips/model_training_dataset/train"
# TRAIN_DATA = "/data/AWT/random_datasets_from_snips/model_training_dataset/val"
VAL_DATA = "/data/AWT/random_datasets_from_snips/model_training_dataset/val"
TEST_DATA = "/data/AWT/random_datasets_from_snips/model_training_dataset/test"
BATCH_SIZE = 60
EPOCHS = 10
LR = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "finetuned_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Correct data transforms with proper channel ordering
train_transform = transforms.Compose([
    transforms.Resize((480, 480)),  # Maintains aspect ratio
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),  # Converts to [C, H, W]
])

val_transform = transforms.Compose([
    transforms.Resize((480, 480)),  # Maintains aspect ratio
    transforms.ToTensor(),  # Converts to [C, H, W]
])

test_transform = transforms.Compose([
    transforms.Resize((480, 480)),  # Maintains aspect ratio
    transforms.ToTensor(),  # Converts to [C, H, W]
])

# Dataset Class with shape validation
class SpeciesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.images = []
        self.transform = transform
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append((os.path.join(cls_dir, img), self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
                # Validate tensor shape
                if img.shape != (3, 480, 480):
                    raise RuntimeError(f"Invalid shape {img.shape} for {img_path}")
            return img, label
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return torch.zeros(3, 480, 480), -1  # Return with correct shape

def sanitize_batch(images, labels):
    """
    Remove any item whose label is -1 or whose tensor is not 3 x 480 x 480.
    Returns None, None if the entire batch is invalid.
    """
    valid_mask = (labels != -1)                       # keep only real labels
    # all tensors already have shape (B, 3, 480, 480) because the Dataset
    # replaced bad images with zeros – so we just rely on the label mask here
    if valid_mask.sum() == 0:
        return None, None

    return images[valid_mask], labels[valid_mask]

# Load data
print("Loading datasets...")
train_dataset = SpeciesDataset(TRAIN_DATA, train_transform)
val_dataset = SpeciesDataset(VAL_DATA, val_transform)
test_dataset = SpeciesDataset(TEST_DATA, test_transform)

# Filter out invalid samples
train_dataset.images = [x for x in train_dataset.images if not x[0].endswith('.error')]
val_dataset.images = [x for x in val_dataset.images if not x[0].endswith('.error')]

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=4,
    pin_memory=True,
    drop_last=False  # Prevents shape errors in last batch ## TODO
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True,
    drop_last=False  # Prevents shape errors in last batch
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=121, 
    shuffle=False, 
    num_workers=2,
    pin_memory=True,
    drop_last=False  # Prevents shape errors in last batch
)

print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Model preparation

# Load base model
print("Preparing model...")
base_classifier = SpeciesNetClassifier(MODEL_NAME)
model = base_classifier.model

# ---------- 1. resize the classifier correctly ----------
NUM_CLASSES = 15
in_feats = model.initializers.onnx_initializer_135.shape[0]   # 1280
model.initializers.onnx_initializer_135 = nn.Parameter(
    torch.empty(in_feats, NUM_CLASSES)         # (1280, 15)  ← correct order
)
model.initializers.onnx_initializer_136 = nn.Parameter(
    torch.zeros(NUM_CLASSES)                   # bias (15,)
)

# ---------- 2. load checkpoint and fix orientation if needed ----------
state_dict_15 = torch.load(MODEL_WEIGHTS, map_location='cpu')

w_key = 'initializers.onnx_initializer_135'
if w_key in state_dict_15:
    w = state_dict_15[w_key]
    if w.shape == (NUM_CLASSES, in_feats):     # (15, 1280) → needs flip
        w = w.t().contiguous()                 # (1280, 15)
    state_dict_15[w_key] = w

# biases are fine – no transpose ever needed
missing, unexpected = model.load_state_dict(state_dict_15, strict=False)

if unexpected:
    print(f"Unexpected keys: {unexpected}")
if missing:
    print(f"Missing keys: {missing}")

# Verify classifier shape
print("Classifier weight shape:", model.initializers.onnx_initializer_135.shape)
print("Classifier bias shape:", model.initializers.onnx_initializer_136.shape)

model = model.to(DEVICE)
model.train()

# Unfreeze all parameters
# for param in model.parameters():
#     param.requires_grad = True

# 0️⃣  freeze everything
for p in model.parameters():
    p.requires_grad = False

# 1️⃣  choose what to train  –  head + top-conv + last MBConv stage
UNFREEZE_PATTERNS = (
    "initializers.onnx_initializer_135",      # classifier.weight  (15 × 1280)
    "initializers.onnx_initializer_136",      # classifier.bias
    "SpeciesNet/efficientnetv2-m/top_conv",   # 1×1 conv before global-pool
    "SpeciesNet/efficientnetv2-m/block7",     # the whole last MBConv stage
)

trainable_params, trainable_names = [], []
for name, p in model.named_parameters():
    if any(name.startswith(pat) for pat in UNFREEZE_PATTERNS):
        p.requires_grad = True
        trainable_params.append(p)
        trainable_names.append(name)

# print("\n✓ 1-stage fine-tune – trainable tensors:")
# for n in trainable_names:
#     print(" •", n)
print("Total trainable params:",
      sum(p.numel() for p in trainable_params))

# 2️⃣  build optimiser *only* from those params
early_stopping_patience = 5  # Stop if no improvement for 5 epochs
early_stopping_counter = 0

# Training setup
nn.init.kaiming_uniform_(model.initializers.onnx_initializer_135, a=math.sqrt(5))
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
# build the optimiser only from the parameters that will train
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR, weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)
best_acc = 0.0

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for images, labels in train_loader:

        # images, labels = sanitize_batch(images, labels)
        # if images is None:   # batch became empty after filtering
        #     continue
        
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        images = images.permute(0, 2, 3, 1).contiguous()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:

            # images, labels = sanitize_batch(images, labels)
            # if images is None:   # batch became empty after filtering
            #     continue
                
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images = images.permute(0, 2, 3, 1).contiguous()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    
    # Calculate metrics
    if total > 0:
        train_loss /= total
        train_acc = correct / total
    else:
        train_loss = 0.0
        train_acc = 0.0
        
    if val_total > 0:
        val_loss /= val_total
        val_acc = val_correct / val_total
    else:
        val_loss = 0.0
        val_acc = 0.0
        
    epoch_time = time.time() - start_time
    
    print(f"\nEpoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s")
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    
    if val_acc > best_acc:
        best_acc = val_acc
        early_stopping_counter = 0  # reset counter
        save_path = os.path.join(SAVE_DIR, f"best_model.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'classes': train_dataset.classes
        }, save_path)
        print(f"Saved best model to {save_path} - {val_acc:.4f}")
    else:
        early_stopping_counter += 1
        print(f"No improvement for {early_stopping_counter} epoch(s)")
        if early_stopping_counter >= early_stopping_patience:
            print(f"\n Early stopping at epoch {epoch+1} (val_acc={val_acc:.4f})")
            break  # exit training loop

    scheduler.step(val_acc)


print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")

# Test phase
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0
all_preds = []
all_labels = []


with torch.no_grad():
    for images, labels in test_loader:

        # images, labels = sanitize_batch(images, labels)
        # if images is None:   # batch became empty after filtering
        #     continue
            
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        images = images.permute(0, 2, 3, 1).contiguous()
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    
if test_total > 0:
    test_loss /= test_total
    test_acc = test_correct / test_total
else:
    test_loss = 0.0
    test_acc = 0.0

# Compute classification metrics
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
plt.close()

print(f"\nTesting complete with accuracy: {test_acc:.4f}")

# #!/usr/bin/env python3
# """
# SpeciesNet Fine-Tuning for 15 Species (Fixed Input Shape)
# """
# import os
# import time
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# from PIL import Image
# from speciesnet.classifier import SpeciesNetClassifier
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# import numpy as np


# # Configuration
# MODEL_NAME = "kaggle:google/speciesnet/pyTorch/v4.0.1a"
# MODEL_WEIGHTS = "speciesnet_15_1init.pth"
# # TRAIN_DATA = "/data/AWT/random_datasets_from_snips/model_training_dataset/train"
# TRAIN_DATA = "/data/AWT/random_datasets_from_snips/model_training_dataset/val"
# VAL_DATA = "/data/AWT/random_datasets_from_snips/model_training_dataset/val"
# TEST_DATA = "/data/AWT/random_datasets_from_snips/model_training_dataset/test"
# BATCH_SIZE = 64
# EPOCHS = 10
# LR = 0.0001
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# SAVE_DIR = "finetuned_models"
# os.makedirs(SAVE_DIR, exist_ok=True)

# # Correct data transforms with proper channel ordering
# train_transform = transforms.Compose([
#     transforms.Resize((480, 480)),  # Maintains aspect ratio
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),  # Converts to [C, H, W]
# ])

# val_transform = transforms.Compose([
#     transforms.Resize((480, 480)),  # Maintains aspect ratio
#     transforms.ToTensor(),  # Converts to [C, H, W]
# ])

# test_transform = transforms.Compose([
#     transforms.Resize((480, 480)),  # Maintains aspect ratio
#     transforms.ToTensor(),  # Converts to [C, H, W]
# ])

# # Dataset Class with shape validation
# class SpeciesDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.classes = sorted(os.listdir(root_dir))
#         self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
#         self.images = []
#         self.transform = transform
        
#         for cls in self.classes:
#             cls_dir = os.path.join(root_dir, cls)
#             for img in os.listdir(cls_dir):
#                 if img.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     self.images.append((os.path.join(cls_dir, img), self.class_to_idx[cls]))
    
#     def __len__(self):
#         return len(self.images)
    
#     def __getitem__(self, idx):
#         img_path, label = self.images[idx]
#         try:
#             img = Image.open(img_path).convert('RGB')
#             if self.transform:
#                 img = self.transform(img)
#                 # Validate tensor shape
#                 if img.shape != (3, 480, 480):
#                     raise RuntimeError(f"Invalid shape {img.shape} for {img_path}")
#             return img, label
#         except Exception as e:
#             print(f"Error loading {img_path}: {str(e)}")
#             return torch.zeros(3, 480, 480), -1  # Return with correct shape

# # Load data
# print("Loading datasets...")
# train_dataset = SpeciesDataset(TRAIN_DATA, train_transform)
# val_dataset = SpeciesDataset(VAL_DATA, val_transform)
# test_dataset = SpeciesDataset(TEST_DATA, test_transform)

# # Filter out invalid samples
# train_dataset.images = [x for x in train_dataset.images if not x[0].endswith('.error')]
# val_dataset.images = [x for x in val_dataset.images if not x[0].endswith('.error')]

# train_loader = DataLoader(
#     train_dataset, 
#     batch_size=60, # BATCH_SIZE
#     shuffle=True, 
#     num_workers=4,
#     pin_memory=True,
#     drop_last=False  # Prevents shape errors in last batch
# )
# val_loader = DataLoader(
#     val_dataset, 
#     batch_size=60, 
#     shuffle=False, 
#     num_workers=2,
#     pin_memory=True,
#     drop_last=False  # Prevents shape errors in last batch
# )
# test_loader = DataLoader(
#     test_dataset, 
#     batch_size=55, 
#     shuffle=False, 
#     num_workers=2,
#     pin_memory=True,
#     drop_last=False  # Prevents shape errors in last batch ## TODO
# )

# print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# def sanitize_batch(images, labels):
#     """
#     Remove any item whose label is -1 or whose tensor is not 3×480×480.
#     Returns None, None if the entire batch is invalid.
#     """
#     valid_mask = (labels != -1)                       # keep only real labels
#     # all tensors already have shape (B, 3, 480, 480) because the Dataset
#     # replaced bad images with zeros – so we just rely on the label mask here
#     if valid_mask.sum() == 0:
#         return None, None

#     return images[valid_mask], labels[valid_mask]

# # Model preparation
# print("Preparing model...")
# base_classifier = SpeciesNetClassifier(MODEL_NAME)
# model = base_classifier.model

# # Load modified weights for 15 classes
# state_dict_15 = torch.load(MODEL_WEIGHTS, map_location='cpu')

# # Find classifier tensor
# classifier_key = None
# for k, v in state_dict_15.items():
#     if v.ndim == 2 and (v.size(0) == 15 or v.size(1) == 15):
#         classifier_key = k
#         print(f"Classifier key: {k}, shape: {v.shape}")
#         break

# # Directly load compatible weights
# model_state_dict = model.state_dict()
# for name, param in state_dict_15.items():
#     # Only load weights with matching names and shapes
#     if name in model_state_dict and param.shape == model_state_dict[name].shape:
#         model_state_dict[name] = param
#         # print(f"Loaded weights for: {name}")
#     else:
#         print(f"Skipped weights for: {name} (shape mismatch)")

# model.load_state_dict(model_state_dict, strict=False)
# model = model.to(DEVICE)
# model.train()

# # Unfreeze all parameters
# # for param in model.parameters():
# #     param.requires_grad = True

# # ─── after `model = model.to(DEVICE)` and *before* you build the optimiser ───

# # 0️⃣  freeze everything
# for p in model.parameters():
#     p.requires_grad = False

# # 1️⃣  choose what to train  –  head + top-conv + last MBConv stage
# UNFREEZE_PATTERNS = (
#     "initializers.onnx_initializer_135",      # classifier.weight  (15 × 1280)
#     "initializers.onnx_initializer_136",      # classifier.bias
#     "SpeciesNet/efficientnetv2-m/top_conv",   # 1×1 conv before global-pool
#     "SpeciesNet/efficientnetv2-m/block7",     # the whole last MBConv stage
# )

# trainable_params, trainable_names = [], []
# for name, p in model.named_parameters():
#     if any(name.startswith(pat) for pat in UNFREEZE_PATTERNS):
#         p.requires_grad = True
#         trainable_params.append(p)
#         trainable_names.append(name)

# # print("\n✓ 1-stage fine-tune – trainable tensors:")
# # for n in trainable_names:
# #     print(" •", n)
# print("Total trainable params:",
#       sum(p.numel() for p in trainable_params))

# # 2️⃣  build optimiser *only* from those params
# optimizer = torch.optim.Adam(trainable_params, lr=LR, weight_decay=1e-4)
# early_stopping_patience = 5  # Stop if no improvement for 5 epochs
# early_stopping_counter = 0

# # Training setup
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='max', patience=3, factor=0.5
# )
# best_acc = 0.0

# # Training loop
# print("Starting training...")
# for epoch in range(EPOCHS):
#     # Training phase
#     model.train()
#     train_loss = 0.0
#     correct = 0
#     total = 0
#     start_time = time.time()
    
#     for images, labels in train_loader:
        
#         images, labels = sanitize_batch(images, labels)
#         if images is None:   # batch became empty after filtering
#             continue
        
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
        
#         images = images.permute(0, 2, 3, 1).contiguous()
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == labels).sum().item()
#         total += labels.size(0)
    
#     # Validation phase
#     model.eval()
#     val_loss = 0.0
#     val_correct = 0
#     val_total = 0
    
#     with torch.no_grad():
#         for images, labels in val_loader:
            
#             images, labels = sanitize_batch(images, labels)
#             if images is None:   # batch became empty after filtering
#                 continue
                
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             images = images.permute(0, 2, 3, 1).contiguous()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             val_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)
#             val_correct += (predicted == labels).sum().item()
#             val_total += labels.size(0)
    
#     # Calculate metrics
#     if total > 0:
#         train_loss /= total
#         train_acc = correct / total
#     else:
#         train_loss = 0.0
#         train_acc = 0.0
        
#     if val_total > 0:
#         val_loss /= val_total
#         val_acc = val_correct / val_total
#     else:
#         val_loss = 0.0
#         val_acc = 0.0
        
#     epoch_time = time.time() - start_time
    
#     print(f"\nEpoch {epoch+1}/{EPOCHS} | Time: {epoch_time:.1f}s")
#     print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
#     print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    
#     if val_acc > best_acc:
#         best_acc = val_acc
#         early_stopping_counter = 0  # reset counter
#         save_path = os.path.join(SAVE_DIR, f"best_model.pth")
#         torch.save({
#             'epoch': epoch+1,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'val_acc': val_acc,
#             'classes': train_dataset.classes
#         }, save_path)
#         print(f"Saved best model to {save_path} - {val_acc:.4f}")
#     else:
#         early_stopping_counter += 1
#         print(f"No improvement for {early_stopping_counter} epoch(s)")
#         if early_stopping_counter >= early_stopping_patience:
#             print(f"\n⛔ Early stopping at epoch {epoch+1} (val_acc={val_acc:.4f})")
#             break  # exit training loop

#     scheduler.step(val_acc)


# print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")

# # Test phase
# model.eval()
# test_loss = 0.0
# test_correct = 0
# test_total = 0
# all_preds = []
# all_labels = []


# with torch.no_grad():
#     for images, labels in test_loader:

#         images, labels = sanitize_batch(images, labels)
#         if images is None:   # batch became empty after filtering
#             continue
            
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#         images = images.permute(0, 2, 3, 1).contiguous()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         test_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs, 1)
#         test_correct += (predicted == labels).sum().item()
#         test_total += labels.size(0)
#         all_preds.extend(predicted.cpu().numpy())
#         all_labels.extend(labels.cpu().numpy())

    
# if test_total > 0:
#     test_loss /= test_total
#     test_acc = test_correct / test_total
# else:
#     test_loss = 0.0
#     test_acc = 0.0

# # Compute classification metrics
# print("\nClassification Report:")
# print(classification_report(all_labels, all_preds, target_names=test_dataset.classes, digits=4))

# # Confusion matrix
# cm = confusion_matrix(all_labels, all_preds)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_dataset.classes)
# fig, ax = plt.subplots(figsize=(12, 12))
# disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', colorbar=False)
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
# plt.close()

# print(f"\nTesting complete with accuracy: {test_acc:.4f}")