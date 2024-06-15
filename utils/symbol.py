from ultralytics import YOLO, RTDETR
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms
import cv2
import os

# replace all torch-10 GELU's by torch-12 GELU
def torchmodify(name) :
    a=name.split('.')
    for i,s in enumerate(a) :
        if s.isnumeric() :
            a[i]="_modules['"+s+"']"
    return '.'.join(a)

def extract_boxes(results):
    """从结果中提取cls, conf, xyxy信息"""
    boxes = results[0].boxes
    return boxes.cls, boxes.conf, boxes.xyxy

def apply_nms_together(cls1, conf1, xyxy1, cls2, conf2, xyxy2, iou_threshold=0.5):
    """合并两个结果并应用NMS"""
    # 合并两个模型的结果
    cls_all = torch.cat([cls1, cls2])
    conf_all = torch.cat([conf1, conf2])
    xyxy_all = torch.cat([xyxy1, xyxy2])

    # 应用NMS
    keep = nms(xyxy_all, conf_all, iou_threshold)
    return cls_all[keep], conf_all[keep], xyxy_all[keep]

def save_results(cls, xyxy, conf, file_name):
    """保存结果到npy文件，使用结构化数组"""
    # 定义结构化数组的数据类型
    dtype = [('symbol', 'U10'), ('bbox', 'i4', (4,)), ('cls_id', 'i4'), ('conf_score', 'f4')]
    results = np.zeros(len(cls), dtype=dtype)
    #count=0
    for i in range(len(cls)):
        symbol = f'symbol_{i+1}'
        bbox = xyxy[i].cpu().numpy().astype('int')
        conf_score = float(conf[i].cpu().numpy())
        #if conf_score>=0.5:
        results[i] = (symbol, bbox, int(cls[i]), conf_score)
            #count+=1

    # 保存结果
    np.save(file_name, results)

def draw_boxes(image_path, boxes, labels, confs, output_path):
    """绘制边界框和标签到图片上"""
    # 加载图片
    image = cv2.imread(image_path)
    
    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    # 绘制每个边界框和标签
    for bbox, label, conf in zip(boxes, labels, confs):
        # 绘制边界框
        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        
        # 准备标签文本
        text = f"{label} {conf:.2f}"
        
        # 获取文本大小
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        # 设置文本背景框
        text_box_position = (int(bbox[0]), int(bbox[1]) - text_size[1])
        text_box_end = (int(bbox[0]) + text_size[0], int(bbox[1]))
        cv2.rectangle(image, text_box_position, text_box_end, (0, 0, 255), cv2.FILLED)
        
        # 绘制文本
        cv2.putText(image, text, (int(bbox[0]), int(bbox[1]) - 5), font, font_scale, (255, 255, 255), font_thickness)

    # 保存或显示图片
    cv2.imwrite(output_path, image)

def symbol_detection(detr_model, yolo_model, image_path):
    """进行符号检测，并保存结果和带有边界框的图片"""
    for name, module in detr_model.named_modules() :
        if isinstance(module,nn.GELU) :
            exec('detr_model.'+torchmodify(name)+'=nn.GELU()')
    # 获取文件名和扩展名
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_path = f"./output/image/{base_name}_symbols.jpg"
    output_npy_path = f"./output/npy/{base_name}_symbols.npy"

    im1 = Image.open(image_path)
    result_detr = detr_model.predict(source=im1, save=False, show_conf=False)
    result_yolo = yolo_model.predict(source=im1, save=False, show_conf=False)

    cls_detr, conf_detr, xyxy_detr = extract_boxes(result_detr)
    cls_yolo, conf_yolo, xyxy_yolo = extract_boxes(result_yolo)

    # 合并结果并应用NMS
    cls_nms, conf_nms, xyxy_nms = apply_nms_together(cls_detr, conf_detr, xyxy_detr, cls_yolo, conf_yolo, xyxy_yolo)
    keep_indices = conf_nms >= 0.5
    cls_nms = cls_nms[keep_indices]
    conf_nms = conf_nms[keep_indices]
    xyxy_nms = xyxy_nms[keep_indices]
    # 保存结果
    save_results(cls_nms, xyxy_nms, conf_nms, file_name=output_npy_path)

    # 绘制边界框并保存图片
    draw_boxes(image_path, xyxy_nms, cls_nms, conf_nms, output_path=output_image_path)
    return output_npy_path

# Example usage:
# detr_model = RTDETR("weights/train11/weights/best.pt")
# yolo_model = YOLO("weights/train4/weights/best.pt")
# symbol_detection(detr_model, yolo_model, "0.jpg")




