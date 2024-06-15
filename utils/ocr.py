import cv2
import numpy as np
from shapely.geometry import Polygon
from mmocr.apis import MMOCRInferencer
import mmcv
import os

# Function to rotate the image 90 degrees clockwise
def rotate_image_90_clockwise(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# Function to extract texts and scores from OCR results
def extract_texts_and_scores(result):
    return [(text, score) for text, score in zip(result['predictions'][0]['rec_texts'], result['predictions'][0]['rec_scores'])]

# Function to rotate bounding box 90 degrees counterclockwise
def rotate_bbox_90_counterclockwise(bbox, image_shape):
    height, width = image_shape[:2]
    rotated_bbox = [(y, width - x) for x, y in bbox]
    return rotated_bbox

# Convert polygon list to tuples
def convert_to_polygon(bbox):
    return [(bbox[i], bbox[i + 1]) for i in range(0, len(bbox), 2)]

# Function to compute IOU between two bounding boxes
def compute_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union

# Function to merge predictions based on IOU and confidence score
def merge_predictions(predictions1, predictions2, iou_threshold=0.2):
    merged_predictions = []
    matched_indices = set()

    for index1, (text1, score1, poly1) in enumerate(predictions1):
        best_match = None
        best_iou = 0
        best_score = score1

        for index2, (text2, score2, poly2) in enumerate(predictions2):
            iou = compute_iou(poly1, poly2)
            if iou >= iou_threshold and iou > best_iou:
                if score2 > best_score:
                    best_match = (text2, score2, poly2)
                    best_score = score2
                    #print("match!")
                else:
                    best_match = (text1, score1, poly1)
                best_iou = iou
                matched_indices.add(index2)

        if best_match is not None:
            merged_predictions.append(best_match)
        else:
            merged_predictions.append((text1, score1, poly1))

    for index2, (text2, score2, poly2) in enumerate(predictions2):
        if index2 not in matched_indices:
            merged_predictions.append((text2, score2, poly2))

    return merged_predictions

# Function to extract texts and scores with polygons and apply score filtering
def extract_texts_and_scores_with_polygons(result, score_threshold=0.7):
    texts = result['predictions'][0]['rec_texts']
    scores = result['predictions'][0]['rec_scores']
    polygons = result['predictions'][0]['det_polygons']
    return [
        (text, score, convert_to_polygon(poly))
        for text, score, poly in zip(texts, scores, polygons)
        if score >= score_threshold and (len(text) > 1 or text.isdigit())
    ]

# Function to draw bounding boxes and annotate text on the image
def draw_bounding_boxes_and_text(image, predictions):
    for text, score, polygon in predictions:
        poly_np = np.array(polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [poly_np], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = np.mean(poly_np, axis=0).astype(int)[0]
        cv2.putText(image, f'{text} ({score:.2f})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return image

# Function to split image into blocks with overlap
def split_image_into_blocks_with_overlap(image, block_size, overlap):
    blocks = []
    img_height, img_width = image.shape[:2]
    step_size = block_size - overlap

    y_positions = range(0, img_height - overlap, step_size)
    x_positions = range(0, img_width - overlap, step_size)

    for y in y_positions:
        for x in x_positions:
            block = image[y:min(y + block_size, img_height), x:min(x + block_size, img_width)]
            blocks.append((block, x, y))
    return blocks

def min_bounding_rectangle(points):
    # 找到最小和最大坐标
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])
    min_y = np.min(points[:, 1])
    max_y = np.max(points[:, 1])

    # 定义最小外接矩形的四个顶点
    rect = np.array([
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
        [max_x, min_y]
    ])

    return rect

# Filter out polygons with any x-coordinate greater than a given threshold
def filter_polygons(predictions, max_x_threshold, min_area):
    filtered_predictions = []
    for text, score, poly in predictions:
        # Check if all x-coordinates are below the threshold
        if all(x <= max_x_threshold for x, y in poly):
            # Calculate the polygon area
            area = Polygon(poly).area
            # Include the bounding box if the area is above the minimum
            if area >= min_area:
                filtered_predictions.append((text, score, poly))
    return filtered_predictions

def refine_merge_predictions(predictions, iou_threshold=0.5, score_diff_threshold=0.1):
    merged = []
    used = set()

    for i, (text1, score1, poly1) in enumerate(predictions):
        if i in used:
            continue

        best_match = (text1, score1, poly1)
        used.add(i)

        for j, (text2, score2, poly2) in enumerate(predictions):
            if j in used:
                continue

            iou = compute_iou(poly1, poly2)
            if iou > iou_threshold:
                # Choose based on score difference
                if abs(score1 - score2) > score_diff_threshold:
                    if score2 > score1:
                        best_match = (text2, score2, poly2)
                        #print("match")
                else:
                    # Otherwise, choose based on text length
                    if len(text2) > len(text1):
                        best_match = (text2, score2, poly2)
                        #print("match")
                    #best_match = (text1, score1, poly1) if len(text1) >= len(text2) else (text2, score2, poly2)

                used.add(j)

        merged.append(best_match)

    return merged
# New function to filter out predictions with bbox area containment
def filter_contained_bboxes(predictions, containment_threshold=0.8):
    filtered_predictions = []
    for i, (text1, score1, poly1) in enumerate(predictions):
        poly1_obj = Polygon(poly1)
        is_contained = False
        for j, (text2, score2, poly2) in enumerate(predictions):
            if i != j:
                poly2_obj = Polygon(poly2)
                intersection = poly1_obj.intersection(poly2_obj).area
                if intersection / poly1_obj.area >= containment_threshold:
                    is_contained = True
                    break
        if not is_contained:
            filtered_predictions.append((text1, score1, poly1))
    return filtered_predictions

def ocr_infer(ocr_model, image_path):
    """Perform OCR inference and save the results."""
    # Load image
    original_image = mmcv.imread(image_path)

    # Rotate the image
    rotated_image = rotate_image_90_clockwise(original_image)

    # Split images into blocks with overlap
    # 1000 200 miss
    # 1000 500 include none exist
    blocks_original = split_image_into_blocks_with_overlap(original_image, 512, 256)
    blocks_rotated = split_image_into_blocks_with_overlap(rotated_image, 512, 256)

    # Perform OCR on original image blocks
    results_original = []
    for block, x_offset, y_offset in blocks_original:
        result = ocr_model(block, show=False, save_vis=False)
        texts_and_scores = extract_texts_and_scores_with_polygons(result)
        adjusted_bboxes = [(text, score, min_bounding_rectangle(np.array([(x + x_offset, y + y_offset) for x, y in poly]))) for text, score, poly in texts_and_scores]
        results_original.extend(adjusted_bboxes)

    # Perform OCR on rotated image blocks
    results_rotated = []
    for block, x_offset, y_offset in blocks_rotated:
        result = ocr_model(block, show=False, save_vis=False)
        texts_and_scores = extract_texts_and_scores_with_polygons(result)
        adjusted_bboxes = [(text, score, min_bounding_rectangle(np.array((rotate_bbox_90_counterclockwise([(x + x_offset, y + y_offset) for x, y in poly], rotated_image.shape))))) for text, score, poly in texts_and_scores]
        results_rotated.extend(adjusted_bboxes)

    # Set thresholds
    max_x_threshold = 5500
    min_area = 330
    filtered_results_original = filter_polygons(results_original, max_x_threshold, min_area)
    filtered_results_rotated = filter_polygons(results_rotated, max_x_threshold, min_area)

    # Merge predictions after filtering
    merged_predictions = merge_predictions(filtered_results_original, filtered_results_rotated)
    merged_predictions = refine_merge_predictions(merged_predictions)
    # Apply the new containment filter
    merged_predictions = filter_contained_bboxes(merged_predictions)

    # Draw bounding boxes and text on the original image
    annotated_image = draw_bounding_boxes_and_text(original_image, merged_predictions)

    # Sort bounding boxes by area (ascending order)
    sorted_bboxes = sorted(merged_predictions, key=lambda x: Polygon(x[2]).area)

    # Print sorted bounding boxes with their areas
    #for text, score, poly in sorted_bboxes:
        #area = Polygon(poly).area
        #print(f'Text: {text}, Score: {score}, Area: {area}')
    npy_data = []
    for i, (text, score, poly) in enumerate(sorted_bboxes):
        bbox_id = f'word_{i + 1}'  # 生成编号
        x0,y0=poly[0]
        x1,y1=poly[2]
        bbox = np.array([x0,y0,x1,y1])
        #import pdb;pdb.set_trace()
        npy_data.append([bbox_id, bbox, text, 0])

    # 转换为 numpy 数组
    npy_array = np.array(npy_data, dtype=object)

    # Save .npy file with the same name as the image file
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    npy_file_name = f"./output/npy/{base_name}_words.npy"
    np.save(npy_file_name, npy_array)

    # Save the final annotated image with the same name as the image file
    output_image_path = f"./output/image/{base_name}_words.jpg"
    cv2.imwrite(output_image_path, annotated_image)
    print(f'Annotated image saved to {output_image_path}')
    print(f'Data saved to {npy_file_name}')
    return npy_file_name

# Example usage:
# ocr_model = MMOCRInferencer(det='DBNetpp', rec='ABINet', device='cuda:0')
# ocr_infer(ocr_model, "0.jpg")

