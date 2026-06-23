## ============================================================
# FINAL AI PROJECT - LEAF IMAGE CLASSIFICATION
# Proposed upgraded architecture: LeafLiteNet-MS
# GroupID: 01
# Framework: PyTorch
# Input:  RGB image tensor (3, 256, 256)
# Output: raw logits with shape (batch_size, num_classes)
# ============================================================
import os
import random
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ============================================================
# 1. CONFIGURATION - CẤU HÌNH CHÍNH
# ============================================================

GROUP_ID = "01"  # Đổi đúng số nhóm nếu nhóm không phải 01
SEED = 42
IMAGE_SIZE = 256  # DO NOT MODIFY: PDF yêu cầu giữ nguyên 256
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
EPOCHS = 50
VAL_RATIO = 0.20
PATIENCE = 5
NUM_WORKERS = 4 # Nên để 0 trên Windows/Visual Studio Code để tránh lỗi multiprocessing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data-clc-classification")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY - CỐ ĐỊNH RANDOM SEED
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Cố định seed để kết quả có thể tái lập."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# 3. DATA TRANSFORMS - TIỀN XỬ LÝ VÀ AUGMENTATION ĐỘNG
#    Lưu ý: Không sửa ảnh gốc, không tạo ảnh mới trong dataset.
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=12),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ============================================================
# 4. DATASET SPLIT - CHIA TRAIN / VALIDATION CÓ STRATIFY
# ============================================================

def stratified_split_indices(targets, val_ratio=0.2, seed=42):
    """Chia dữ liệu theo từng class để train/validation không lệch lớp."""
    targets = np.array(targets)
    train_indices = []
    val_indices = []
    generator = torch.Generator().manual_seed(seed)

    for class_id in np.unique(targets):
        class_indices = np.where(targets == class_id)[0]
        perm = torch.randperm(len(class_indices), generator=generator).numpy()
        class_indices = class_indices[perm]

        val_count = max(1, int(len(class_indices) * val_ratio))
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    random.Random(seed).shuffle(train_indices)
    random.Random(seed).shuffle(val_indices)
    return train_indices, val_indices


def build_dataloaders():
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Không tìm thấy dataset tại: {DATA_DIR}\n"
            "Hãy đặt thư mục data-clc-classification cùng cấp với file code .py."
        )

    base_dataset = datasets.ImageFolder(root=DATA_DIR)
    class_names = base_dataset.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Number of classes:", num_classes)
    print("Total images:", len(base_dataset))

    if num_classes != 4:
        print("Cảnh báo: Dataset dự kiến có 4 lớp Class01, Class02, Class03, Class04.")

    train_indices, val_indices = stratified_split_indices(
        base_dataset.targets, val_ratio=VAL_RATIO, seed=SEED
    )

    train_dataset_full = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # Class weights để giảm ảnh hưởng mất cân bằng dữ liệu.
    targets = np.array(base_dataset.targets)
    train_targets = targets[train_indices]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Class counts in train split:", class_counts.tolist())
    print("Class weights:", class_weights.detach().cpu().numpy().round(4).tolist())

    return train_loader, val_loader, class_names, num_classes, class_weights


# ============================================================
# 5. LEAFLITENET-MS ARCHITECTURE
#    Mô hình được giữ tên BasicCNN để phát triển từ starter code.
# ============================================================

class ConvBNAct(nn.Sequential):
    """Conv + BatchNorm + ReLU: khối CNN cơ bản."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Module):
    """
    Khối Depthwise Separable Convolution.
    Ý nghĩa nâng cấp:
    - Depthwise Conv học đặc trưng không gian riêng từng kênh.
    - Pointwise Conv 1x1 trộn thông tin giữa các kênh.
    - Giảm số tham số so với convolution thường.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    """
    SE Attention Block - Squeeze-and-Excitation.
    Ý nghĩa nâng cấp:
    - Học trọng số chú ý theo channel.
    - Tăng kênh đặc trưng quan trọng, giảm kênh ít liên quan.
    - Phù hợp ảnh lá vì dấu hiệu bệnh thường nằm ở một số kênh màu/texture nhất định.
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        weights = self.pool(x).view(batch_size, channels)
        weights = self.fc(weights).view(batch_size, channels, 1, 1)
        return x * weights


class MultiScaleBlock(nn.Module):
    """
    Multi-Scale Block.
    Ý nghĩa nâng cấp:
    - Nhánh 3x3 học đốm nhỏ, cạnh nhỏ, texture nhỏ.
    - Nhánh 5x5 học vùng bệnh lớn hơn, vệt dài, mảng cháy rộng.
    - Concatenation ghép hai góc nhìn.
    - 1x1 Conv Fusion trộn và nén kênh.
    - SE Attention chọn kênh quan trọng.
    """
    def __init__(self, in_channels, out_channels, stride=2, use_se=True):
        super().__init__()
        branch1_channels = out_channels // 2
        branch2_channels = out_channels - branch1_channels

        self.branch3 = DepthwiseSeparableConv(
            in_channels, branch1_channels, kernel_size=3, stride=stride
        )
        self.branch5 = DepthwiseSeparableConv(
            in_channels, branch2_channels, kernel_size=5, stride=stride
        )

        self.fuse = ConvBNAct(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()

    def forward(self, x):
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x = torch.cat([x3, x5], dim=1)
        x = self.fuse(x)
        x = self.se(x)
        return x


class ResidualDSBlock(nn.Module):
    """
    Residual Depthwise Block.
    Ý nghĩa nâng cấp:
    - Học F(x), sau đó cộng lại với input x: output = F(x) + x.
    - Giữ thông tin cũ như texture/gân lá.
    - Cải thiện luồng gradient, giúp huấn luyện ổn định hơn.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x) + x)


class BasicCNN(nn.Module):
    """
    LeafLiteNet-MS: Lightweight Multi-Scale CNN for Leaf Image Classification.

    Input : (batch_size, 3, 256, 256)
    Output: (batch_size, num_classes) raw logits

    Không dùng Softmax/LogSoftmax trong forward vì CrossEntropyLoss xử lý logits.
    """
    def __init__(self, num_classes=4, dropout_rate=0.30):
        super().__init__()

        # Stem Conv Block: trích xuất cạnh, màu sắc và texture ban đầu.
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2)
        # Output: 32 x 128 x 128

        # Depthwise Separable Conv: giảm tham số, tăng hiệu quả tính toán.
        self.ds1 = DepthwiseSeparableConv(32, 48, kernel_size=3, stride=2)
        # Output: 48 x 64 x 64

        # Multi-Scale Block 1: học đốm nhỏ và vùng bệnh lớn ở mức trung bình.
        self.ms1 = MultiScaleBlock(48, 80, stride=2, use_se=True)
        # Output: 80 x 32 x 32

        # Residual Block: giữ thông tin cũ và ổn định gradient.
        self.res1 = ResidualDSBlock(80)
        # Output: 80 x 32 x 32

        # Multi-Scale Block 2 + SE: học đặc trưng bệnh lá cấp cao hơn.
        self.ms2 = MultiScaleBlock(80, 112, stride=2, use_se=True)
        # Output: 112 x 16 x 16

        # Depthwise Separable Conv cuối: nén không gian, tăng kênh đặc trưng.
        self.ds2 = DepthwiseSeparableConv(112, 160, kernel_size=3, stride=2)
        # Output: 160 x 8 x 8

        # Dual Global Pooling:
        # GAP nhìn tổng thể lá; GMP bắt vùng bệnh nổi bật nhất.
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # Dropout chống overfitting.
        self.dropout = nn.Dropout(p=dropout_rate)

        # Classifier xuất raw logits cho 4 lớp, tuyệt đối không Softmax.
        self.classifier = nn.Linear(160 * 2, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.ds1(x)
        x = self.ms1(x)
        x = self.res1(x)
        x = self.ms2(x)
        x = self.ds2(x)

        avg_feature = self.gap(x).flatten(1)
        max_feature = self.gmp(x).flatten(1)
        x = torch.cat([avg_feature, max_feature], dim=1)

        x = self.dropout(x)
        x = self.classifier(x)

        # Output raw logits: shape = (batch_size, num_classes)
        return x


# ============================================================
# 6. UTILITY FUNCTIONS
# ============================================================

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_model_validity(model: nn.Module, num_classes: int) -> None:
    """Kiểm tra các điều kiện kỹ thuật quan trọng trước khi train/export."""
    model.eval()
    total_params = count_trainable_parameters(model)
    print("Trainable parameters:", total_params)

    if total_params >= 100000:
        raise ValueError("Mô hình vượt giới hạn 100,000 trainable parameters.")

    with torch.no_grad():
        dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
        output = model(dummy)

    print("Dummy output shape:", tuple(output.shape))
    if output.shape != (2, num_classes):
        raise ValueError("Output shape sai. Yêu cầu: (batch_size, num_classes).")


# ============================================================
# 7. TRAINING AND EVALUATION LOOPS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# ============================================================
# 8. PLOTS AND REPORT FILES
# ============================================================

def save_training_curves(history: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Training Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_acc"], label="Training Accuracy")
    plt.plot(history["epoch"], history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300)
    plt.close()


def save_confusion_matrix(y_true, y_pred, class_names) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():
    print("Device:", DEVICE)
    print("Project directory:", PROJECT_DIR)
    print("Dataset directory:", DATA_DIR)
    print("Output directory:", OUTPUT_DIR)

    train_loader, val_loader, class_names, num_classes, class_weights = build_dataloaders()

    model = BasicCNN(num_classes=num_classes, dropout_rate=0.30).to(DEVICE)
    check_model_validity(model, num_classes)

    # CrossEntropyLoss phù hợp bài toán multi-class classification.
    # Output model là raw logits, không Softmax.
    try:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    except TypeError:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW: có weight_decay giúp regularization, giảm overfitting.
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Scheduler giúp learning rate giảm mềm theo epoch, hỗ trợ hội tụ ổn định.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    history_records = []
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "learning_rate": current_lr,
        })

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}"
        )

        # Chọn best model theo val accuracy, nếu hòa thì ưu tiên val loss thấp hơn.
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping tại epoch {epoch}. Best Val Acc = {best_val_acc:.4f}")
            break

    # Load lại best state trước khi đánh giá cuối và export .pt
    model.load_state_dict(best_state)
    model.to(DEVICE)

    history = pd.DataFrame(history_records)
    history.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)
    save_training_curves(history)

    val_loss, val_acc, y_true, y_pred = evaluate(model, val_loader, criterion)

    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n========== FINAL VALIDATION METRICS ==========")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision Macro: {precision_macro:.4f}")
    print(f"Recall Macro   : {recall_macro:.4f}")
    print(f"F1-score Macro : {f1_macro:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )
    pd.DataFrame(report_dict).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "classification_report_validation.csv")
    )

    save_confusion_matrix(y_true, y_pred, class_names)

    total_params = count_trainable_parameters(model)
    print("\n========== MODEL SUMMARY ==========")
    print("Architecture: LeafLiteNet-MS inside class BasicCNN")
    print("Trainable parameters:", total_params)
    print("Input shape : (batch_size, 3, 256, 256)")
    print("Output shape: (batch_size, num_classes)")
    print("Output type : raw logits, no Softmax")

    if total_params >= 100000:
        raise ValueError("Mô hình không hợp lệ vì vượt 100,000 trainable parameters.")

    # ========================================================
    # 10. TORCHSCRIPT EXPORT - DO NOT MODIFY THIS SECTION
    # ========================================================
    ############################################
    # DO NOT MODIFY THIS SECTION
    ############################################
    model.eval()
    example_input = torch.randn(1,3,IMAGE_SIZE,IMAGE_SIZE).to(DEVICE)
    traced_model = torch.jit.trace(model,example_input)
    GroupID = "01"
    model_name = f"{GroupID}_DeepLearningProject_TrainedModel.pt"
    traced_model.save(model_name)
    print("Model saved:", model_name)

    # Kiểm tra nhanh file .pt có load được bằng torch.jit.load hay không.
    loaded_model = torch.jit.load(model_name, map_location=DEVICE)
    loaded_model.eval()
    with torch.no_grad():
        test_output = loaded_model(example_input)
    print("TorchScript load test output shape:", tuple(test_output.shape))


if __name__ == "__main__":
    main()
