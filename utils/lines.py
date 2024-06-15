import os
import numpy as np
import cv2
import torch
from sklearn.cluster import DBSCAN
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as lsc


def classify_lines(pred_lines):
    horizontal_lines = []
    vertical_lines = []
    for line in pred_lines:
        (x0, y0), (x1, y1) = line
        if abs(y1 - y0) < abs(x1 - x0):  # 判断为水平线
            horizontal_lines.append(line)
        else:  # 判断为竖直线
            vertical_lines.append(line)
    return np.array(horizontal_lines), np.array(vertical_lines)

def sort_and_group_lines(lines, threshold=10, direction='horizontal'):
    if len(lines) == 0:
        return []
    
    if direction == 'horizontal':
        # 根据线段的中点 y 坐标排序
        lines = sorted(lines, key=lambda line: np.mean([line[0][1], line[1][1]]))
    else:  # vertical
        # 根据线段的中点 x 坐标排序
        lines = sorted(lines, key=lambda line: np.mean([line[0][0], line[1][0]]))
    
    grouped_lines = []
    current_group = [lines[0]]
    
    for line in lines[1:]:
        if direction == 'horizontal':
            # 计算当前线段的中点 y 坐标
            current_midpoint = np.mean([line[0][1], line[1][1]])
            # 计算当前组最后一条线段的中点 y 坐标
            last_midpoint = np.mean([current_group[-1][0][1], current_group[-1][1][1]])
        else:  # vertical
            # 计算当前线段的中点 x 坐标
            current_midpoint = np.mean([line[0][0], line[1][0]])
            # 计算当前组最后一条线段的中点 x 坐标
            last_midpoint = np.mean([current_group[-1][0][0], current_group[-1][1][0]])
        
        # 判断是否属于同一组
        if abs(current_midpoint - last_midpoint) < threshold:
            current_group.append(line)
        else:
            grouped_lines.append(current_group)
            current_group = [line]
    grouped_lines.append(current_group)
    
    return grouped_lines

def merge_lines(lines, labels):
    unique_labels = set(labels)
    merged_lines = []
    for label in unique_labels:
        if label == -1:
            continue
        label_lines = [line for line, lbl in zip(lines, labels) if lbl == label]
        x_coords = np.concatenate([[line[0][0], line[1][0]] for line in label_lines])
        y_coords = np.concatenate([[line[0][1], line[1][1]] for line in label_lines])
        if abs(y_coords.ptp()) < abs(x_coords.ptp()):  # 判断为水平线
            x0, x1 = x_coords.min(), x_coords.max()
            y = y_coords.mean()
            merged_lines.append([[x0, y], [x1, y]])
        else:  # 判断为竖直线
            y0, y1 = y_coords.min(), y_coords.max()
            x = x_coords.mean()
            merged_lines.append([[x, y0], [x, y1]])
    return np.array(merged_lines)

def is_overlapping(line_a, line_b, threshold=0.8):
    (x0_a, y0_a), (x1_a, y1_a) = line_a
    (x0_b, y0_b), (x1_b, y1_b) = line_b
    if abs(y1_a - y0_a) < abs(x1_a - x0_a):  # 水平线
        overlap = max(0, min(x1_a, x1_b) - max(x0_a, x0_b))
        length_a = x1_a - x0_a
        return overlap / length_a > threshold
    else:  # 竖直线
        overlap = max(0, min(y1_a, y1_b) - max(y0_a, y0_b))
        length_a = y1_a - y0_a
        return overlap / length_a > threshold
    
def merge_overlapping_segments(merged_lines):
    merged = []
    used = [False] * len(merged_lines)
    for i, line_a in enumerate(merged_lines):
        if used[i]:
            continue
        x0_a, y0_a = line_a[0]
        x1_a, y1_a = line_a[1]
        for j, line_b in enumerate(merged_lines[i+1:], start=i+1):
            if used[j]:
                continue
            x0_b, y0_b = line_b[0]
            x1_b, y1_b = line_b[1]
            if is_overlapping(line_a, line_b, threshold=0.1):  # 使用更低的阈值判断是否合并
                if abs(y1_a - y0_a) < abs(x1_a - x0_a):  # 水平线
                    x0 = min(x0_a, x0_b)
                    x1 = max(x1_a, x1_b)
                    y = (y0_a + y1_a + y0_b + y1_b) / 4
                    line_a = [[x0, y], [x1, y]]
                else:  # 竖直线
                    y0 = min(y0_a, y0_b)
                    y1 = max(y1_a, y1_b)
                    x = (x0_a + x1_a + x0_b + x1_b) / 4
                    line_a = [[x, y0], [x, y1]]
                used[j] = True
        merged.append(line_a)
    return merged

def dbscan_clustering(grouped_lines, eps=100, min_samples=1):
    clustered_lines = []
    for lines in grouped_lines:
        if len(lines) == 0:
            continue
        mid_points = np.array([(line[0] + line[1]) / 2 for line in lines])
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(mid_points)
        labels = clustering.labels_
        merged_lines = merge_lines(lines, labels)
        # 双向判断线段是否覆盖并进行筛选
        keep = [True] * len(merged_lines)
        for i, line_a in enumerate(merged_lines):
            for j, line_b in enumerate(merged_lines):
                if i != j and (keep[i] or keep[j]) and is_overlapping(line_a, line_b):
                    if is_overlapping(line_a, line_b):
                        if abs(line_a[1][0] - line_a[0][0]) > abs(line_b[1][0] - line_b[0][0]) or \
                           abs(line_a[1][1] - line_a[0][1]) > abs(line_b[1][1] - line_b[0][1]):
                            keep[j] = False
                            break
                        else:
                            keep[i] = False
                            break
        
        filtered_merged_lines = [line for line, k in zip(merged_lines, keep) if k]
        merged_filtered_lines = merge_overlapping_segments(filtered_merged_lines)
        
        clustered_lines.append(merged_filtered_lines)
    return clustered_lines

def merge_clusters(horizontal_clusters, vertical_clusters):
    merged_lines = []
    for cluster in horizontal_clusters + vertical_clusters:
        merged_lines.extend(cluster)
    return np.array(merged_lines)


def merge_lines_dbscan(lines, eps=45, min_samples=1):
    line_features = []
    for line in lines:
        try:
            pt1, pt2 = line[0][:2], line[0][2:]
            dx, dy = pt2 - pt1
        except:
            pt1, pt2 = line[0], line[1]
            dx, dy = pt2 - pt1
        length = np.linalg.norm([dx, dy])
        angle = np.arctan2(dy, dx)
        mid_point = (pt1 + pt2) / 2
        line_features.append([mid_point[0], mid_point[1], angle, length])
    
    line_features = np.array(line_features)
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(line_features)
    labels = clustering.labels_
    
    merged_lines = []
    
    for label in np.unique(labels):
        cluster_indices = np.where(labels == label)[0]
        if len(cluster_indices) > 1:
            points = np.concatenate([lines[i][0].reshape(-1, 2) for i in cluster_indices], axis=0)
            min_pt = points.min(axis=0)
            max_pt = points.max(axis=0)
            merged_lines.append([min_pt[0], min_pt[1], max_pt[0], max_pt[1]])
    
    return merged_lines

def merge_lines1(lines, orientation):
    if not lines:
        return []

    def custom_sort_key(line):
        if orientation == 'horizontal':
            y_key = round(line[1] / 3)
            x_key = round(line[0] / 3)
            return (y_key, x_key)
        else:  # 'vertical'
            x_key = round(line[0] / 3)
            y_key = round(line[1] / 3)
            return (x_key, y_key)

    lines.sort(key=custom_sort_key)
    merged_lines = []
    current_group = [lines[0]]

    for current_line in lines[1:]:
        last_line = current_group[-1]
        can_merge = False

        if orientation == 'horizontal':
            if (abs(current_line[1] - last_line[1]) < 10 and 
                abs(current_line[0] - last_line[0]) < 100 and
                abs(current_line[2] - last_line[2]) < 100):
                can_merge = True
        else:
            if (abs(current_line[0] - last_line[0]) < 10 and
                abs(current_line[1] - last_line[1]) < 100 and
                abs(current_line[3] - last_line[3]) < 100):
                can_merge = True

        if can_merge:
            current_group.append(current_line)
        else:
            if current_group:
                merged_line = merge_line_group(current_group, orientation)
                merged_lines.append(merged_line)
            current_group = [current_line]

    if current_group:
        merged_line = merge_line_group(current_group, orientation)
        merged_lines.append(merged_line)
    
    return merged_lines

def merge_line_group(lines, orientation):
    if orientation == 'horizontal':
        y_avg = sum(line[1] for line in lines) / len(lines)
        x_min = min(line[0] for line in lines)
        x_max = max(line[2] for line in lines)
        return [x_min, y_avg, x_max, y_avg]
    else:
        x_avg = sum(line[0] for line in lines) / len(lines)
        y_min = min(line[1] for line in lines)
        y_max = max(line[3] for line in lines)
        return [x_avg, y_min, x_avg, y_max]

def classify_and_merge_lines(lines):
    horizontal_lines = [line for line in lines if abs(line[0] - line[2]) > abs(line[1] - line[3])]
    vertical_lines = [line for line in lines if abs(line[0] - line[2]) <= abs(line[1] - line[3])]
    merged_horizontal_lines = merge_lines1(horizontal_lines, 'horizontal')
    merged_vertical_lines = merge_lines1(vertical_lines, 'vertical')
    return merged_horizontal_lines + merged_vertical_lines

def detect_and_merge_lines(gray_image_path,img_path, model,device):
    img = cv2.imread(gray_image_path)[:, :, ::-1]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    scale_factor = 1 / 3
    width = int(gray_img.shape[1] * scale_factor)
    height = int(gray_img.shape[0] * scale_factor)
    dim = (width, height)

    resized_img = cv2.resize(gray_img, dim, interpolation=cv2.INTER_AREA)

    model = model.to(device).eval()

    inputs = {'image': torch.tensor(resized_img, dtype=torch.float, device=device)[None, None] / 255.}
    with torch.no_grad():
        out = model(inputs)
        pred_lines = out['lines'][0]

    temp_l = []
    for dline in pred_lines:
        x0, y0 = dline[0]
        x1, y1 = dline[1]
        if ((400 * scale_factor <= x0 <= 5500 * scale_factor and 300 * scale_factor <= y0 <= 4320 * scale_factor) or 
            (400 * scale_factor <= x1 <= 5500 * scale_factor and 300 * scale_factor <= y1 <= 4320 * scale_factor)):
            temp_l.append(dline)

    pred_lines = np.array(temp_l)
    pred_lines = pred_lines / scale_factor
    horizontal_lines, vertical_lines = classify_lines(pred_lines)

    # 对水平线段进行排序和分组
    grouped_h_lines = sort_and_group_lines(horizontal_lines,direction='horizontal')

    # 对竖直线段进行排序和分组
    grouped_v_lines = sort_and_group_lines(vertical_lines,direction='vertical')

    # 对分组后的水平线段进行 DBSCAN 聚类
    h_clusters = dbscan_clustering(grouped_h_lines)

    # 对分组后的竖直线段进行 DBSCAN 聚类
    v_clusters = dbscan_clustering(grouped_v_lines)

    # 合并聚类结果
    pred_lines = merge_clusters(h_clusters, v_clusters)
    output_image = gray_img.copy()
    for line in pred_lines:
        pt1 = (int(line[0][0]), int(line[0][1]))
        pt2 = (int(line[1][0]), int(line[1][1]))
        cv2.line(output_image, pt1, pt2, (255, 255, 255), thickness=15)

    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(output_image)[0]

    filtered_dlines = []
    for dline in lines:
        x0, y0, x1, y1 = map(int, map(round, dline[0]))
        if (400 <= x0 <= 5500 and 300 <= y0 <= 4320) or (400 <= x1 <= 5500 and 300 <= y1 <= 4320):
            filtered_dlines.append(dline)

    merged_lines = merge_lines_dbscan(filtered_dlines, eps=10)
    merged_lines = classify_and_merge_lines(merged_lines)
    temp_=[]
    for line in merged_lines:
        x0, y0,x1, y1 = line
        # 计算线段长度
        length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        if length>=30:
            temp_.append(line)
    merged_lines=temp_
    all_lines = []
    line_counter = 1
    for line in pred_lines:
        pt1 = [int(line[0][0]), int(line[0][1])]
        pt2 = [int(line[1][0]), int(line[1][1])]
        all_lines.append([f'line_{line_counter}', list(pt1 + pt2), '1', 'solid'])
        line_counter += 1

    for line in merged_lines:
        pt1 = [int(line[0]), int(line[1])]
        pt2 = [int(line[2]), int(line[3])]
        all_lines.append([f'line_{line_counter}', list(pt1 + pt2), '1', 'dashed'])
        line_counter += 1
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_npy_path = f"./output/npy/{base_name}_lines.npy"
    output_img_path = f"./output/image/{base_name}_lines.jpg"
    np.save(output_npy_path, np.array(all_lines, dtype=object))
    output_with_dashed_lines = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in all_lines:
            pt1 = (int(line[1][0]), int(line[1][1]))
            pt2 = (int(line[1][2]), int(line[1][3]))
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.line(output_with_dashed_lines, pt1, pt2, color, thickness=2)
    cv2.imwrite(output_img_path, output_with_dashed_lines)

    return output_npy_path

# Example usage:
# image = cv2.imread('0_bw.jpg')
# ckpt = 'weights/deeplsd_md.tar'
# ckpt = torch.load(str(ckpt), map_location='cpu')
# conf = {
#     'detect_lines': True,  # Whether to detect lines or only DF/AF
#     'line_detection_params': {
#         'merge': True,  # Whether to merge close-by lines
#         'filtering': 'strict',  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
#         'grad_thresh': 4,
#         'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
#     }
# }
# model = DeepLSD(conf)
# model.load_state_dict(ckpt['model'])
# output_image_path = detect_and_merge_lines('gray.jpg',"114514.jpg", model)
# print(output_image_path)
