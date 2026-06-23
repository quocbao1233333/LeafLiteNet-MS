# 01_DeepLearningProject - LeafLiteNet-MS

## 1. Giới thiệu

Dự án này xây dựng mô hình Deep Learning cho bài toán phân loại ảnh lá cây 4 lớp bằng PyTorch. Kiến trúc được đề xuất là **LeafLiteNet-MS**, một mô hình CNN nhẹ được nâng cấp từ starter code `BasicCNN` theo hướng hiệu quả, ít tham số và phù hợp yêu cầu Final Project.

Mục tiêu chính:

- Phân loại ảnh vào 4 lớp: `Class01`, `Class02`, `Class03`, `Class04`.
- Input cố định: ảnh RGB kích thước `3 × 256 × 256`.
- Output: raw logits dạng `(batch_size, 4)`.
- Tổng số trainable parameters nhỏ hơn `100,000`.
- Xuất mô hình TorchScript `.pt` đúng định dạng yêu cầu.

## 2. Kiến trúc mô hình

Mô hình chính vẫn được triển khai trong:

```python
class BasicCNN(nn.Module):
```

Bên trong lớp này, kiến trúc được nâng cấp thành **LeafLiteNet-MS** với các khối chính:

1. **Stem Conv Block**: trích xuất đặc trưng ban đầu từ ảnh RGB.
2. **Depthwise Separable Conv Block**: giảm số tham số so với convolution thường.
3. **Multi-Scale Block**: học đặc trưng ở nhiều kích thước, phù hợp với đốm nhỏ và vùng bệnh lớn trên lá.
4. **Residual Block**: giữ thông tin đặc trưng cũ và cải thiện luồng gradient.
5. **SE Attention Block**: nhấn mạnh các kênh đặc trưng quan trọng.
6. **Global Average Pooling + Global Max Pooling**: kết hợp đặc trưng tổng thể và đặc trưng nổi bật nhất.
7. **Dropout**: giảm overfitting.
8. **Linear Classifier**: xuất 4 raw logits, không dùng Softmax trong model.

## 3. Cấu trúc thư mục

Đặt project trong Visual Studio Code theo cấu trúc sau:

```text
D:\Visualstudiocode_Projects\01_DeepLearningProject_LeafClassification
│
├── 01_DeepLearningProject_Code.py
│
├── data-clc-classification
│   ├── Class01
│   ├── Class02
│   ├── Class03
│   └── Class04
│
└── outputs
```

Lưu ý: không đổi tên folder class, không thêm ảnh ngoài, không xóa ảnh, không chỉnh sửa ảnh gốc trong dataset.

## 4. Thư viện sử dụng

Các nhóm thư viện chính:

```python
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
```

Cài đặt thư viện cần thiết:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## 5. Cách chạy chương trình

Mở Terminal trong Visual Studio Code tại thư mục project, sau đó chạy:

```bash
python 01_DeepLearningProject_Code.py
```

Khi chạy, chương trình sẽ:

1. Kiểm tra đường dẫn project và dataset.
2. Đọc dữ liệu bằng `ImageFolder`.
3. Chia dữ liệu thành train/validation.
4. Khởi tạo mô hình LeafLiteNet-MS.
5. Đếm trainable parameters.
6. Huấn luyện mô hình.
7. Lưu log, biểu đồ và confusion matrix vào thư mục `outputs`.
8. Xuất mô hình TorchScript `.pt`.

## 6. Các file đầu ra

Sau khi chạy thành công, chương trình sẽ tạo:

```text
01_DeepLearningProject_TrainedModel.pt
outputs/training_log.csv
outputs/loss_curve.png
outputs/accuracy_curve.png
outputs/classification_report_validation.csv
outputs/confusion_matrix.png
```

File cần nộp cuối cùng theo đúng quy định:

```text
01_DeepLearningProject_Code.py
01_DeepLearningProject_TrainedModel.pt
01_DeepLearningProject_Report.pdf
```

## 7. Các yêu cầu kỹ thuật đã đảm bảo

- `IMAGE_SIZE = 256` được giữ nguyên.
- Input model là ảnh RGB dạng `(3, 256, 256)`.
- Output model là `(batch_size, 4)`.
- Output cuối là raw logits.
- Không dùng Softmax hoặc LogSoftmax trong model.
- Không trả về tuple hoặc dictionary trong `forward()`.
- Trainable parameters nhỏ hơn `100,000`.
- Dataset gốc không bị chỉnh sửa.
- Augmentation chỉ thực hiện động trong `transforms.Compose()`.
- File model được export bằng TorchScript `.pt`.
- File `.pt` có thể load bằng `torch.jit.load()`.

## 8. Huấn luyện và đánh giá

Chương trình sử dụng:

- Loss function: `CrossEntropyLoss`.
- Optimizer: `AdamW`.
- Scheduler: `CosineAnnealingLR`.
- Early Stopping để giảm overfitting.
- Class weights để hỗ trợ dữ liệu mất cân bằng nhẹ.

Các chỉ số được theo dõi:

- Training loss.
- Validation loss.
- Training accuracy.
- Validation accuracy.
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion Matrix.

## 9. Ghi chú quan trọng

Validation set chỉ dùng để đánh giá nội bộ và tối ưu mô hình. Test set chính thức là test ẩn do giảng viên giữ riêng. Vì vậy kết quả validation không thay thế cho điểm chấm cuối cùng trên test ẩn.

Không chỉnh sửa các phần sau:

- Không đổi `IMAGE_SIZE = 256`.
- Không sửa cấu trúc dataset.
- Không thêm Softmax ở cuối model.
- Không sửa sai khối export TorchScript.
- Không để model vượt quá `100,000` trainable parameters.
- Không nộp sai tên file.

## 10. Tác giả / Nhóm

GroupID: `01`

Project: Deep Learning Image Classification with LeafLiteNet-MS
