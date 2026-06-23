# ============================================================
# TENSORFLOW REFERENCE VERSION - LEAF IMAGE CLASSIFICATION
# Architecture: LeafLiteNet-MS converted from PyTorch to TensorFlow/Keras
#
# =================================================================
# CANH BAO QUAN TRONG
# =================================================================
# File nay chi dung de HOC / DOI CHIEU voi PyTorch.
# Neu de bai final project yeu cau PyTorch + TorchScript .pt thi KHONG NOP file nay.
#
# Ly do:
# - TensorFlow/Keras luu model dang .keras hoac SavedModel.
# - De bai PyTorch thuong yeu cau file .pt load bang torch.jit.load().
#
# Input TensorFlow/Keras:
#   (batch_size, 256, 256, 3)
#
# Output:
#   (batch_size, num_classes)
#   la RAW LOGITS, khong phai xac suat, khong Softmax trong model.
# ============================================================

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
# 1. CONFIGURATION - CAU HINH CHINH
# ============================================================
# Cac bien duoi day la sieu tham so va duong dan chinh cua chuong trinh.

GROUP_ID = "01"          # Ma nhom. Neu nhom khac 01 thi doi lai cho dung.
SEED = 42                # Seed co dinh de ket qua co the lap lai.
IMAGE_SIZE = 256         # Kich thuoc anh dau vao: 256x256.
BATCH_SIZE = 64          # So anh dua vao model trong moi lan cap nhat trong so.
LEARNING_RATE = 5e-4     # Toc do hoc ban dau.
EPOCHS = 50              # So epoch toi da.
VAL_RATIO = 0.20         # Ty le validation: 20%.
PATIENCE = 5             # Early stopping neu val_accuracy khong tang sau 5 epoch.
AUTOTUNE = tf.data.AUTOTUNE

# PROJECT_DIR la thu muc chua file .py nay.
PROJECT_DIR = Path(__file__).resolve().parent

# Dataset phai dat cung cap voi file .py:
# PROJECT_DIR/
#   01_DeepLearningProject_Code_TensorFlow_COMMENTED.py
#   data-clc-classification/
#       Class01/
#       Class02/
#       Class03/
#       Class04/
DATA_DIR = PROJECT_DIR / "data-clc-classification"

# Thu muc luu log, bieu do, confusion matrix, model TensorFlow.
OUTPUT_DIR = PROJECT_DIR / "outputs_tensorflow"
OUTPUT_DIR.mkdir(exist_ok=True)

# Mean/std ImageNet.
# Muc dich: dua pixel ve phan phoi on dinh hon, giup model train nhanh va de hoi tu.
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


# ============================================================
# 2. REPRODUCIBILITY - CO DINH RANDOM SEED
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Co dinh cac nguon random.
    Nhiem vu:
    - Giam sai khac ket qua giua cac lan chay.
    - Giu viec chia train/validation va khoi tao model on dinh hon.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


set_seed(SEED)


# ============================================================
# 3. DATASET UTILS - DOC DATASET VA CHIA TRAIN/VAL STRATIFIED
# ============================================================
# TensorFlow khong dung torchvision.datasets.ImageFolder.
# Vi vay ta tu doc duong dan anh trong cac folder class.
#
# Cau truc dataset mong doi:
# data-clc-classification/
#   Class01/
#       img_001.jpg
#       ...
#   Class02/
#   Class03/
#   Class04/

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(data_dir: Path):
    """
    Doc tat ca anh va nhan class tu cau truc folder.

    Input:
        data_dir: thu muc data-clc-classification

    Output:
        image_paths: danh sach duong dan anh
        labels: nhan so tu 0 den num_classes-1
        class_names: ten class lay tu ten folder, vi du Class01, Class02...

    Ban chat:
        Folder nao chua anh thi ten folder do chinh la label/class.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Khong tim thay dataset tai: {data_dir}\n"
            "Hay dat thu muc data-clc-classification cung cap voi file .py."
        )

    # Sap xep ten folder de label on dinh:
    # Class01 -> 0, Class02 -> 1, Class03 -> 2, Class04 -> 3
    class_names = sorted([
        p.name for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ])

    image_paths = []
    labels = []

    for class_id, class_name in enumerate(class_names):
        class_dir = data_dir / class_name

        # rglob("*") doc tat ca file ben trong class folder.
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
                image_paths.append(str(path))
                labels.append(class_id)

    return np.array(image_paths), np.array(labels, dtype=np.int32), class_names


def stratified_split_indices(labels, val_ratio=0.2, seed=42):
    """
    Chia train/validation theo tung class.

    Nhiem vu:
    - Dam bao validation khong bi lech class.
    - Moi class deu co anh trong train va validation.

    Vi du:
    Class01 co 1000 anh, val_ratio=0.2
    -> 200 anh vao validation, 800 anh vao training.
    """
    train_indices = []
    val_indices = []
    rng = np.random.default_rng(seed)

    labels = np.array(labels)

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]

        # Xao tron rieng tung class de chia cong bang.
        rng.shuffle(class_indices)

        val_count = max(1, int(len(class_indices) * val_ratio))

        # Lay phan dau lam validation, phan con lai lam training.
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    # Xao tron tong the sau khi ghep cac class.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return np.array(train_indices), np.array(val_indices)


# ============================================================
# 4. PREPROCESSING + AUGMENTATION DONG
# ============================================================

def decode_resize_normalize(path, label):
    """
    Doc anh tu duong dan, resize, chuyen sang float, normalize.

    Cac buoc:
    1. tf.io.read_file: doc file anh thanh bytes.
    2. tf.io.decode_image: giai ma thanh tensor RGB.
    3. tf.image.resize: dua anh ve 256x256.
    4. chia 255: dua pixel tu [0,255] ve [0,1].
    5. normalize: (image - mean) / std.

    Output:
        image co shape (256, 256, 3)
        label la so class.
    """
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])

    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32) / 255.0

    image = (image - MEAN) / STD
    return image, label


def augment_image(image, label):
    """
    Tang cuong du lieu dong trong RAM, khong tao file anh moi.

    Nhiem vu:
    - Lam du lieu train da dang hon.
    - Giam overfitting.
    - Giu dataset goc khong bi thay doi.

    Cac phep dung o day:
    - random_flip_left_right: lat ngang anh la.
    - random_brightness: thay doi do sang.
    - random_contrast: thay doi tuong phan.
    - random_saturation: thay doi do bao hoa mau.
    - random_hue: thay doi sac do mau nhe.

    Luu y:
    Anh dau vao ham nay da normalize, nen ta dua ve [0,1] truoc khi jitter mau,
    sau do normalize lai.
    """
    image = image * STD + MEAN
    image = tf.clip_by_value(image, 0.0, 1.0)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.image.random_saturation(image, lower=0.85, upper=1.15)
    image = tf.image.random_hue(image, max_delta=0.03)
    image = tf.clip_by_value(image, 0.0, 1.0)

    image = (image - MEAN) / STD
    return image, label


def make_dataset(paths, labels, training: bool):
    """
    Tao tf.data.Dataset.

    training=True:
        - shuffle anh
        - preprocess
        - augmentation
        - batch
        - prefetch

    training=False:
        - preprocess
        - batch
        - prefetch
        - KHONG augmentation vi validation phai on dinh.
    """
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(decode_resize_normalize, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def build_datasets():
    """
    Ham tong hop:
    - Doc danh sach anh.
    - Chia train/validation.
    - Tinh class_weight de giam anh huong mat can bang du lieu.
    - Tao train_ds va val_ds.
    """
    image_paths, labels, class_names = list_image_files(DATA_DIR)
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Number of classes:", num_classes)
    print("Total images:", len(image_paths))

    if num_classes != 4:
        print("Canh bao: Dataset du kien co 4 lop Class01, Class02, Class03, Class04.")

    train_indices, val_indices = stratified_split_indices(labels, VAL_RATIO, SEED)

    train_paths = image_paths[train_indices]
    train_labels = labels[train_indices]
    val_paths = image_paths[val_indices]
    val_labels = labels[val_indices]

    # Class weights:
    # Class it anh hon se co trong so cao hon trong loss.
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights_array = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))
    class_weight = {i: float(w) for i, w in enumerate(class_weights_array)}

    print("Train images:", len(train_paths))
    print("Validation images:", len(val_paths))
    print("Class counts in train split:", class_counts.tolist())
    print("Class weights:", np.round(class_weights_array, 4).tolist())

    train_ds = make_dataset(train_paths, train_labels, training=True)
    val_ds = make_dataset(val_paths, val_labels, training=False)

    return train_ds, val_ds, class_names, num_classes, class_weight


# ============================================================
# 5. LEAFLITENET-MS ARCHITECTURE - TENSORFLOW/KERAS
# ============================================================
# Keras dung channel_last:
#   PyTorch:     (batch, channel, height, width)
#   TensorFlow:  (batch, height, width, channel)
#
# Vi vay shape trong comment TensorFlow ghi:
#   256 x 256 x 3
# thay vi:
#   3 x 256 x 256


def conv_bn_act(x, out_channels, kernel_size=3, stride=1, name=None):
    """
    CONV-BN-RELU BLOCK

    Nhiem vu:
    - Conv2D: hoc dac trung cuc bo cua anh.
    - BatchNorm: on dinh phan phoi feature map.
    - ReLU: tao phi tuyen, giup model hoc quan he phuc tap.

    Dung trong:
    - Stem Conv Block
    - 1x1 Conv Fusion sau khi concatenate trong Multi-Scale Block.

    Thong so:
    - out_channels: so kenh dau ra.
    - kernel_size: kich thuoc kernel.
    - stride: neu stride=2 thi giam kich thuoc H,W di mot nua.
    """
    x = layers.Conv2D(
        out_channels,
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        name=None if name is None else name + "_conv",
    )(x)

    x = layers.BatchNormalization(name=None if name is None else name + "_bn")(x)
    x = layers.ReLU(name=None if name is None else name + "_relu")(x)
    return x


def depthwise_separable_conv(x, out_channels, kernel_size=3, stride=1, name=None):
    """
    DEPTHWISE SEPARABLE CONVOLUTION BLOCK

    Day la khoi nang cap quan trong giup model nhe hon Conv2D thuong.

    Gom 2 buoc:

    1. DepthwiseConv2D:
       - Moi input channel duoc tich chap rieng.
       - Hoc dac trung khong gian: canh la, van la, texture, dom benh.
       - Khong tron thong tin giua cac channel.

    2. Pointwise Conv2D 1x1:
       - Tron thong tin giua cac channel.
       - Doi so kenh tu input channels thanh out_channels.

    Loi ich:
    - Giam tham so.
    - Giam chi phi tinh toan.
    - Van giu kha nang hoc dac trung anh.
    """
    # Buoc 1: Depthwise convolution
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        name=None if name is None else name + "_dw",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_dw_bn")(x)
    x = layers.ReLU(name=None if name is None else name + "_dw_relu")(x)

    # Buoc 2: Pointwise convolution 1x1
    x = layers.Conv2D(
        out_channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=None if name is None else name + "_pw",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else name + "_pw_bn")(x)
    x = layers.ReLU(name=None if name is None else name + "_pw_relu")(x)

    return x


def se_block(x, channels, reduction=8, name=None):
    """
    SE ATTENTION BLOCK - SQUEEZE AND EXCITATION

    Nhiem vu:
    - Hoc trong so quan trong cho tung channel.
    - Channel nao chua dau hieu benh quan trong se duoc nhan manh.
    - Channel it lien quan se bi giam anh huong.

    Luong xu ly:
    1. Squeeze:
       GlobalAveragePooling2D bien H x W x C thanh vector C.
       Moi gia tri dai dien cho muc do kich hoat trung binh cua 1 channel.

    2. Excitation:
       Dense -> ReLU -> Dense -> Sigmoid.
       Tao ra vector trong so tu 0 den 1 cho C channels.

    3. Scale:
       Nhan feature map ban dau voi vector trong so.
       Output van giu nguyen shape H x W x C.
    """
    hidden = max(channels // reduction, 4)

    # Squeeze: H x W x C -> C
    s = layers.GlobalAveragePooling2D(name=None if name is None else name + "_gap")(x)

    # Excitation: C -> hidden -> C
    s = layers.Dense(hidden, activation="relu", name=None if name is None else name + "_fc1")(s)
    s = layers.Dense(channels, activation="sigmoid", name=None if name is None else name + "_fc2")(s)

    # Reshape de nhan lai voi feature map H x W x C
    s = layers.Reshape((1, 1, channels), name=None if name is None else name + "_reshape")(s)

    # Scale channel-wise attention
    x = layers.Multiply(name=None if name is None else name + "_scale")([x, s])
    return x


def multi_scale_block(x, out_channels, stride=2, use_se=True, name=None):
    """
    MULTI-SCALE BLOCK

    Nhiem vu:
    - Hoc dac trung o nhieu kich thuoc khac nhau.
    - Phu hop bai toan la cay vi vet benh co the nho, lon, dai, loang mau.

    Cac nhanh ben trong:

    Branch A - DSConv 3x3:
        Bat chi tiet nho:
        - dom nho
        - canh nho
        - texture min
        - vet benh nho tren la

    Branch B - DSConv 5x5:
        Bat vung rong hon:
        - mang chay la
        - vet benh lon
        - vung mau bat thuong lon

    Concatenate:
        Ghep dac trung tu 2 nhanh theo chieu channel.

    1x1 Conv Fusion:
        Tron dac trung sau khi ghep.
        Dua so channel ve out_channels.

    SE Attention:
        Chon channel quan trong sau khi fusion.
    """
    branch1_channels = out_channels // 2
    branch2_channels = out_channels - branch1_channels

    # Branch A: kernel 3x3
    b3 = depthwise_separable_conv(
        x,
        branch1_channels,
        kernel_size=3,
        stride=stride,
        name=None if name is None else name + "_branch3",
    )

    # Branch B: kernel 5x5
    b5 = depthwise_separable_conv(
        x,
        branch2_channels,
        kernel_size=5,
        stride=stride,
        name=None if name is None else name + "_branch5",
    )

    # Ghep hai nhanh theo chieu channel.
    x = layers.Concatenate(axis=-1, name=None if name is None else name + "_concat")([b3, b5])

    # Fusion bang 1x1 Conv: tron channel va giu out_channels.
    x = conv_bn_act(
        x,
        out_channels,
        kernel_size=1,
        stride=1,
        name=None if name is None else name + "_fuse",
    )

    # SE Attention de nhan manh channel quan trong.
    if use_se:
        x = se_block(x, out_channels, reduction=8, name=None if name is None else name + "_se")

    return x


def residual_ds_block(x, channels, name=None):
    """
    RESIDUAL DEPTHWISE SEPARABLE BLOCK

    Nhiem vu:
    - Hoc them dac trung moi F(x).
    - Giu lai dac trung cu x bang skip connection.
    - Output = F(x) + x.

    Tai sao can residual?
    - Giam mat thong tin khi di qua nhieu layer.
    - Giup gradient truyen nguoc tot hon.
    - Giu cac dac trung quan trong cua la: gan la, texture, mau nen.

    Dieu kien cong duoc:
    - F(x) va x phai co cung shape.
    - Trong block nay stride=1 va channels khong doi, nen cong truc tiep duoc.
    """
    shortcut = x

    # Main path: hoc F(x)
    y = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        name=None if name is None else name + "_dw",
    )(x)
    y = layers.BatchNormalization(name=None if name is None else name + "_dw_bn")(y)
    y = layers.ReLU(name=None if name is None else name + "_dw_relu")(y)

    y = layers.Conv2D(
        channels,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=None if name is None else name + "_pw",
    )(y)
    y = layers.BatchNormalization(name=None if name is None else name + "_pw_bn")(y)

    # Skip connection: output = F(x) + x
    x = layers.Add(name=None if name is None else name + "_add")([shortcut, y])
    x = layers.ReLU(name=None if name is None else name + "_out_relu")(x)

    return x


def build_leaflitenet_ms(num_classes=4, dropout_rate=0.30):
    """
    XAY DUNG TOAN BO MODEL LEAFLITENET-MS

    Luong shape TensorFlow:
    Input:
        256 x 256 x 3

    Stem:
        128 x 128 x 32

    DSConv Block 1:
        64 x 64 x 48

    Multi-Scale Block 1:
        32 x 32 x 80

    Residual DS Block:
        32 x 32 x 80

    Multi-Scale Block 2:
        16 x 16 x 112

    DSConv Block 2:
        8 x 8 x 160

    Dual Global Pooling:
        GAP -> 160 features
        GMP -> 160 features
        Concat -> 320 features

    Classifier:
        Dense 320 -> 4 raw logits
    """
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")

    # --------------------------------------------------------
    # BLOCK 1: STEM CONV BLOCK
    # --------------------------------------------------------
    # Muc dich:
    # - Chuyen anh RGB thanh 32 feature maps dau tien.
    # - Giam kich thuoc tu 256x256 xuong 128x128.
    # - Hoc dac trung co ban: canh la, mau sac, texture ban dau.
    x = conv_bn_act(inputs, 32, kernel_size=3, stride=2, name="stem")
    # Output: 128 x 128 x 32

    # --------------------------------------------------------
    # BLOCK 2: DEPTHWISE SEPARABLE CONV BLOCK 1
    # --------------------------------------------------------
    # Muc dich:
    # - Tiep tuc giam kich thuoc xuong 64x64.
    # - Tang channel tu 32 len 48.
    # - Hoc dac trung tot hon nhung it tham so hon Conv2D thuong.
    x = depthwise_separable_conv(x, 48, kernel_size=3, stride=2, name="ds1")
    # Output: 64 x 64 x 48

    # --------------------------------------------------------
    # BLOCK 3: MULTI-SCALE BLOCK 1
    # --------------------------------------------------------
    # Muc dich:
    # - Branch 3x3 bat chi tiet nho.
    # - Branch 5x5 bat vung benh rong hon.
    # - Concat + 1x1 fusion tron thong tin.
    # - SE Attention chon channel quan trong.
    x = multi_scale_block(x, 80, stride=2, use_se=True, name="ms1")
    # Output: 32 x 32 x 80

    # --------------------------------------------------------
    # BLOCK 4: RESIDUAL DS BLOCK
    # --------------------------------------------------------
    # Muc dich:
    # - Giu nguyen shape 32x32x80.
    # - Hoc F(x) roi cong lai voi input x.
    # - Giu thong tin cu va giup train on dinh.
    x = residual_ds_block(x, 80, name="res1")
    # Output: 32 x 32 x 80

    # --------------------------------------------------------
    # BLOCK 5: MULTI-SCALE BLOCK 2
    # --------------------------------------------------------
    # Muc dich:
    # - Hoc dac trung benh la cap cao hon.
    # - Giam kich thuoc tu 32x32 xuong 16x16.
    # - Tang channel tu 80 len 112.
    x = multi_scale_block(x, 112, stride=2, use_se=True, name="ms2")
    # Output: 16 x 16 x 112

    # --------------------------------------------------------
    # BLOCK 6: DEPTHWISE SEPARABLE CONV BLOCK 2
    # --------------------------------------------------------
    # Muc dich:
    # - Nen khong gian xuong 8x8.
    # - Tang channel len 160 de chua nhieu dac trung cap cao.
    # - Chuan bi feature map cho global pooling.
    x = depthwise_separable_conv(x, 160, kernel_size=3, stride=2, name="ds2")
    # Output: 8 x 8 x 160

    # --------------------------------------------------------
    # BLOCK 7: DUAL GLOBAL POOLING
    # --------------------------------------------------------
    # GAP:
    # - Lay gia tri trung binh tren moi feature map.
    # - Dai dien dac trung tong the cua la.
    #
    # GMP:
    # - Lay gia tri lon nhat tren moi feature map.
    # - Giu dau hieu noi bat nhat, vi du dom benh ro nhat.
    #
    # Concat:
    # - GAP 160 + GMP 160 = 320 features.
    avg_feature = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    max_feature = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Concatenate(axis=-1, name="dual_pool_concat")([avg_feature, max_feature])
    # Output: 320 features

    # --------------------------------------------------------
    # BLOCK 8: DROPOUT
    # --------------------------------------------------------
    # Muc dich:
    # - Chong overfitting.
    # - Trong training, ngau nhien tat 30% feature.
    # - Khi validation/test, Dropout tu dong tat.
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # --------------------------------------------------------
    # BLOCK 9: LINEAR CLASSIFIER
    # --------------------------------------------------------
    # Dense 320 -> num_classes.
    # activation=None nghia la output la raw logits, KHONG phai xac suat.
    #
    # Neu muon xem xac suat khi test rieng thi dung Softmax BEN NGOAI model:
    # probs = tf.nn.softmax(logits, axis=1)
    outputs = layers.Dense(num_classes, activation=None, name="classifier_logits")(x)
    # Output: 4 raw logits

    model = keras.Model(inputs=inputs, outputs=outputs, name="LeafLiteNet_MS_TensorFlow")
    return model


# ============================================================
# 6. PLOTS AND REPORT FILES
# ============================================================

def save_training_curves(history_df: pd.DataFrame):
    """
    Luu 2 bieu do:
    1. Training loss va validation loss.
    2. Training accuracy va validation accuracy.

    Cac hinh nay dung dua vao bao cao de chung minh qua trinh training that.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["loss"], label="Training Loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history_df["epoch"], history_df["accuracy"], label="Training Accuracy")
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png", dpi=300)
    plt.close()


def evaluate_and_save_metrics(model, val_ds, class_names):
    """
    Danh gia model tren validation set.

    Output tinh:
    - Accuracy
    - Precision macro
    - Recall macro
    - F1 macro
    - Classification report tung class
    - Confusion matrix

    Luu y:
    Model tra raw logits.
    Khi chon class du doan:
        argmax(logits)
    Khong can Softmax de lay class.
    """
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        logits = model.predict(images, verbose=0)
        preds = np.argmax(logits, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n========== FINAL VALIDATION METRICS ==========")
    print(f"Accuracy       : {acc:.4f}")
    print(f"Precision Macro: {precision_macro:.4f}")
    print(f"Recall Macro   : {recall_macro:.4f}")
    print(f"F1-score Macro : {f1_macro:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        OUTPUT_DIR / "classification_report_validation.csv"
    )

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    return acc, precision_macro, recall_macro, f1_macro


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():
    """
    Chuong trinh chinh.

    Thu tu chay:
    1. In thong tin moi truong.
    2. Doc dataset va tao train/validation pipeline.
    3. Xay dung LeafLiteNet-MS TensorFlow.
    4. Kiem tra parameter va output shape.
    5. Compile model voi loss from_logits=True.
    6. Train model.
    7. Luu log, bieu do, confusion matrix.
    8. Luu model .keras.
    """
    print("TensorFlow version:", tf.__version__)
    print("Project directory:", PROJECT_DIR)
    print("Dataset directory:", DATA_DIR)
    print("Output directory:", OUTPUT_DIR)

    train_ds, val_ds, class_names, num_classes, class_weight = build_datasets()

    model = build_leaflitenet_ms(num_classes=num_classes, dropout_rate=0.30)

    # Kiem tra so tham so.
    total_params = model.count_params()
    print("Trainable parameters:", total_params)

    if total_params >= 100000:
        raise ValueError("Mo hinh vuot gioi han 100,000 trainable parameters.")

    # Kiem tra output shape voi dummy input.
    dummy = tf.random.normal((2, IMAGE_SIZE, IMAGE_SIZE, 3))
    dummy_output = model(dummy, training=False)
    print("Dummy output shape:", tuple(dummy_output.shape))

    if tuple(dummy_output.shape) != (2, num_classes):
        raise ValueError("Output shape sai. Yeu cau: (batch_size, num_classes).")

    # SparseCategoricalCrossentropy:
    # - Dung cho multi-class classification voi label dang so nguyen 0,1,2,3.
    # - from_logits=True vi classifier khong co Softmax.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # AdamW neu TensorFlow ho tro.
    # Neu ban TensorFlow cu khong co AdamW thi fallback ve Adam.
    try:
        optimizer = keras.optimizers.AdamW(
            learning_rate=LEARNING_RATE,
            weight_decay=1e-4,
        )
    except AttributeError:
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        # EarlyStopping:
        # Neu validation accuracy khong tang sau PATIENCE epoch thi dung.
        # restore_best_weights=True giup lay lai model tot nhat.
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),

        # ReduceLROnPlateau:
        # Neu val_loss khong giam, learning rate se giam mot nua.
        # Giup model hoi tu on dinh hon.
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Bat dau huan luyen.
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Luu training log ra CSV.
    history_df = pd.DataFrame(history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(OUTPUT_DIR / "training_log.csv", index=False)

    # Luu bieu do loss/accuracy.
    save_training_curves(history_df)

    # Danh gia cuoi cung tren validation set.
    evaluate_and_save_metrics(model, val_ds, class_names)

    print("\n========== MODEL SUMMARY ==========")
    print("Architecture: LeafLiteNet-MS TensorFlow/Keras")
    print("Trainable parameters:", model.count_params())
    print("Input shape : (batch_size, 256, 256, 3)")
    print("Output shape: (batch_size, num_classes)")
    print("Output type : raw logits, no Softmax")

    # Luu model TensorFlow/Keras.
    # Luu y: file .keras KHONG thay the duoc .pt neu de bai yeu cau PyTorch.
    keras_model_name = f"{GROUP_ID}_DeepLearningProject_TrainedModel_TensorFlow.keras"
    keras_model_path = PROJECT_DIR / keras_model_name
    model.save(keras_model_path)
    print("TensorFlow/Keras model saved:", keras_model_path)

    # Kiem tra load lai model.
    loaded_model = keras.models.load_model(keras_model_path)
    test_output = loaded_model(dummy, training=False)
    print("Keras load test output shape:", tuple(test_output.shape))


if __name__ == "__main__":
    main()
