import cv2
import numpy as np

def remove_boxes(image_path, boxes_paths):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 遍历所有包含盒子坐标的文件
    for box_path in boxes_paths:
        # 加载坐标数据
        boxes = np.load(box_path, allow_pickle=True)
        
        # 遍历所有盒子，boxes可能包含多个元素，每个元素类似于 ['word_248', [6760, 4271, 6788, 4309], '3', 0]
        for box in boxes:
            x1,y1,x2,y2=box[1] 
            y1,y2=min(y1,y2),max(y1,y2)
            x1,x2=min(x1,x2),max(x1,x2)
            x1,x2,y1,y2=map(int,[x1,x2,y1,y2])
            image[y1:y2, x1:x2] = [255, 255, 255]

    
    # 保存处理后的图像
    return image

# # 图像路径和盒子坐标文件的路径
# image_path = '499.jpg'  # 替换为你的图片路径
# boxes_paths = ['499_words.npy', '499_symbols.npy']  # 替换为你的.npy文件路径

# remove_boxes(image_path, boxes_paths)
