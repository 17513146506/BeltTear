import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import ImageDraw, ImageFont
import os

class BeltClassificationNet(nn.Module):
    def __init__(self):
        super(BeltClassificationNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def predict(image_path, seg_model, class_model, transform):
    # Perform segmentation
    results = seg_model(image_path)
    seg_image = Image.fromarray(results[0].orig_img)

    # Prepare image for classification
    if transform:
        seg_image = transform(seg_image)
    seg_image = seg_image.unsqueeze(0).to(device)

    # Perform classification
    class_model.eval()
    with torch.no_grad():
        output = class_model(seg_image)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = output.max(1)

    class_prediction = predicted.item()
    class_probability = probabilities[0][class_prediction].item() * 100

    # Get defect positions from segmentation results
    defect_positions = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            x, y, w, h = box.xywh[0]
            defect_positions.append([x.item(), y.item(), w.item(), h.item()])

    return class_prediction, class_probability, defect_positions

def visualize_and_save_results(image_path, class_prediction, class_probability, defect_positions, output_path):
    # Open the original image
    original_image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(original_image)

    # Try to find a suitable font
    font = None
    font_size = 30
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        # If default font fails, try some common font paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Common on Linux
            "/System/Library/Fonts/Helvetica.ttc",  # Common on macOS
            "C:\\Windows\\Fonts\\arial.ttf",  # Common on Windows
            "arial.ttf",  # In the current directory
        ]
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue

    if font is None:
        print("Warning: Could not load any font. Text may not be displayed correctly.")

    # Draw bounding boxes for defects
    for pos in defect_positions:
        x, y, width, height = pos
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2
        draw.rectangle([left, top, right, bottom], outline="red", width=3)

    # Add classification probability text
    text = f"{'Defect' if class_prediction == 1 else 'No Defect'}: {class_probability:.2f}%"
    for pos in defect_positions:
        x, y, width, height = pos
        draw.text((x, y - height / 2 - 10), text, fill="blue", font=font)

    # Create a figure and display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(original_image)

    # Add title with class prediction
    plt.title(text, fontsize=16)

    # Remove axis ticks
    plt.axis('off')

    # Save the figure
    output_path = os.path.splitext(output_path)[0] + '.png'
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Result image saved to {output_path}")

# Load segmentation model
seg_model = YOLO('./best.pt')

# Load classification model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BeltClassificationNet().to(device)
model.load_state_dict(torch.load('belt_classification_model.pth', map_location=device))

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Example usage of predict function and visualization
# 修改主函数
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict defects in belt images")
    parser.add_argument("--test_folder", type=str, default="./test", help="Path to the folder containing test images")
    parser.add_argument("--output_folder", type=str, default="./results", help="Path to save the result images")
    args = parser.parse_args()

    # 确保输出文件夹存在
    os.makedirs(args.output_folder, exist_ok=True)

    # 加载模型
    seg_model = YOLO('./best.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeltClassificationNet().to(device)
    model.load_state_dict(torch.load('belt_classification_model.pth', map_location=device))

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 遍历测试文件夹中的所有图片
    for filename in os.listdir(args.test_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(args.test_folder, filename)
            output_path = os.path.join(args.output_folder, f"result_{filename}")

            print(f"Processing image: {filename}")

            # 预测
            class_prediction, class_probability, defect_positions = predict(image_path, seg_model, model, transform)

            print(f"  Class prediction: {'Defect' if class_prediction == 1 else 'No Defect'}")
            print(f"  Classification probability: {class_probability:.2f}%")
            if defect_positions:
                print("  Defect positions:")
                for pos in defect_positions:
                    print(f"    x: {pos[0]}, y: {pos[1]}, width: {pos[2]}, height: {pos[3]}")
            else:
                print("  No defects detected.")

            # 可视化并保存结果
            visualize_and_save_results(image_path, class_prediction, class_probability, defect_positions, output_path)

            print(f"  Result image saved to {output_path}")
            print()

    print("All images processed.")