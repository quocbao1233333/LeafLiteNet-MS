import torch
from PIL import Image
from torchvision import transforms

# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================
MODEL_PATH = "01_DeepLearningProject_TrainedModel.pt"

# Đổi đường dẫn này thành ảnh bạn muốn test
IMAGE_PATH = r"D:\Visualstudiocode_Projects\01_DeepLearningProject_LeafClassification\data-clc-classification\Class01\class01_0001.jpg"
IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Thứ tự class phải đúng như ImageFolder đã đọc
CLASS_NAMES = ["Class01", "Class02", "Class03", "Class04"]

# =========================
# TRANSFORM GIỐNG VALIDATION
# =========================
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# LOAD MODEL TORCHSCRIPT
# =========================
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# =========================
# LOAD ẢNH
# =========================
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = test_transform(image)
input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  # thêm batch dimension

# =========================
# DỰ ĐOÁN
# =========================
with torch.no_grad():
    logits = model(input_tensor)          # raw logits
    probabilities = torch.softmax(logits, dim=1)  # chỉ dùng ngoài model để xem xác suất
    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[0, predicted_index].item()

print("Ảnh test:", IMAGE_PATH)
print("Logits:", logits.cpu().numpy())
print("Xác suất từng class:")

for class_name, prob in zip(CLASS_NAMES, probabilities[0].cpu().numpy()):
    print(f"{class_name}: {prob * 100:.2f}%")

print("--------------------------------")
print("Dự đoán cuối cùng:", predicted_class)
print(f"Độ tin cậy: {confidence * 100:.2f}%")