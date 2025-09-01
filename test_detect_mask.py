from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path


def process_images(model_path, input_folder, output_folder, conf_thres=0.25):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 加载模型
    model = YOLO(model_path)

    # 获取输入文件夹中的所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_folder) if Path(f).suffix.lower() in image_extensions]

    # 定义颜色方案 (BGR格式)
    mask_color = (255, 165, 0)  # 深橙色
    text_bg_color = (0, 69, 255)  # 红色
    text_color = (255, 255, 255)  # 白色

    # 处理每张图片
    for img_file in image_files:
        # 构建完整的输入输出路径
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, f'result_{img_file}')

        # 读取图像
        image = cv2.imread(input_path)
        if image is None:
            print(f"无法读取图像: {input_path}")
            continue

        # 进行预测
        results = model.predict(image, conf=conf_thres, save=False)

        # 获取第一个结果
        result = results[0]

        # 创建一个用于叠加的图像副本
        overlay = image.copy()

        # 如果有检测到目标
        if result.masks is not None:
            # 获取masks和置信度
            masks = result.masks.data.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            # 处理每个检测结果
            for mask, conf, box in zip(masks, confidences, boxes):
                # 将mask转换为图像大小
                mask = mask.astype(np.uint8)
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

                # 创建彩色mask
                colored_mask = np.zeros_like(image)
                colored_mask[mask > 0.5] = mask_color

                # 将mask叠加到图像上
                cv2.addWeighted(colored_mask, 0.5, overlay, 1, 0, overlay)

                # 获取边界框坐标用于放置标签
                x1, y1 = int(box[0]), int(box[1])

                # 准备标签文本
                label = f"QingWeiSiLie {conf:.2f}"

                # 获取文本大小
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)

                # 绘制文本背景
                padding = 5
                cv2.rectangle(overlay,
                              (x1, y1 - text_height - 2 * padding),
                              (x1 + text_width + 2 * padding, y1),
                              text_bg_color,
                              -1)

                # 绘制文本
                cv2.putText(overlay,
                            label,
                            (x1 + padding, y1 - padding),
                            cv2.FONT_HERSHEY_DUPLEX,
                            0.7,
                            text_color,
                            2)

        # 保存结果图像
        cv2.imwrite(output_path, overlay)
        print(f"处理完成: {img_file}")


if __name__ == "__main__":
    # 配置路径
    model_path = 'runs/segment/train18/weights/best.pt'
    input_folder = 'detection_input'
    output_folder = 'results'

    # 设置置信度阈值
    confidence_threshold = 0.25

    # 处理图像
    process_images(model_path, input_folder, output_folder, confidence_threshold)
    print("所有图像处理完成！")