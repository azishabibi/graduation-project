import numpy as np
import cv2
from PIL import Image
import random
import os 
# Function to calculate Euclidean distance between two points
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Function to calculate the minimum distance between a point and a rectangle
def point_to_rectangle_distance(point, rectangle):
    x_point, y_point = point
    xmin, ymin, xmax, ymax = rectangle
    if x_point < xmin:
        x_nearest = xmin
    elif x_point > xmax:
        x_nearest = xmax
    else:
        x_nearest = x_point
    
    if y_point < ymin:
        y_nearest = ymin
    elif y_point > ymax:
        y_nearest = ymax
    else:
        y_nearest = y_point
    
    return distance((x_point, y_point), (x_nearest, y_nearest))

def bbox_min_distance(bbox1, bbox2):
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    if x1_max < x2_min:
        dx = x2_min - x1_max
    elif x2_max < x1_min:
        dx = x1_min - x2_max
    else:
        dx = 0

    if y1_max < y2_min:
        dy = y2_min - y1_max
    elif y2_max < y1_min:
        dy = y1_min - y2_max
    else:
        dy = 0

    return np.sqrt(dx**2 + dy**2)

# Function to calculate the shortest distance between two line segments
def line_segment_distance(line1, line2):
    def point_line_distance(point, line):
        px, py = point
        x1, y1, x2, y2 = line
        line_mag = distance((x1, y1), (x2, y2))
        if line_mag == 0:
            return distance(point, (x1, y1))
        
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_mag**2
        if u < 0:
            closest_point = (x1, y1)
        elif u > 1:
            closest_point = (x2, y2)
        else:
            closest_point = (x1 + u * (x2 - x1), y1 + u * (y2 - y1))
        
        return distance(point, closest_point)
    
    d1 = point_line_distance(line1[:2], line2)
    d2 = point_line_distance(line1[2:], line2)
    d3 = point_line_distance(line2[:2], line1)
    d4 = point_line_distance(line2[2:], line1)
    
    return min(d1, d2, d3, d4)

# Function to check if two lines intersect
def do_lines_intersect(line1, line2):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    A, B = line1[:2], line1[2:]
    C, D = line2[:2], line2[2:]
    
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

# Function to check adjacency between two lines based on their endpoints or intersection
def check_line_adjacency(line1, line2, threshold):
    if do_lines_intersect(line1, line2):
        return True
    return line_segment_distance(line1, line2) <= threshold

# Function to check association between a line and a symbol
def check_line_symbol_association(line, symbol_box, threshold):
    line_endpoints = [line[:2], line[2:]]
    symbol_rect = symbol_box
    #import pdb;pdb.set_trace()
    x_point_min, y_point_min = min(line_endpoints, key=lambda x: x[0])[0], min(line_endpoints, key=lambda x: x[1])[1]
    x_point_max, y_point_max = max(line_endpoints, key=lambda x: x[0])[0], max(line_endpoints, key=lambda x: x[1])[1]
    xmin, ymin, xmax, ymax = symbol_rect

    if abs(y_point_min - y_point_max) <= threshold:
        if ymin <= y_point_min <= ymax and not (x_point_max < xmin or x_point_min > xmax):
            #print(symbol_box,line)
            return True

    if abs(x_point_min - x_point_max) <= threshold:
        if xmin <= x_point_min <= xmax and not (y_point_max < ymin or y_point_min > ymax):
            return True

    for point in line_endpoints:
        if point_to_rectangle_distance(point, symbol_rect) <= threshold:
            return True

    return False

# Function to check if a text box is inside a symbol or intersects it
def check_text_symbol_connection(text_box, symbol_box):
    tx_min, ty_min, tx_max, ty_max = text_box
    sx_min, sy_min, sx_max, sy_max = symbol_box

    if sx_min <= tx_min <= tx_max <= sx_max and sy_min <= ty_min <= ty_max <= sy_max:
        return True

    return False

# Function to find the closest symbol or line for a text box
def connect_text_to_closest_entity(text_box, symbols, lines):
    closest_distance = float('inf')
    closest_entity = None
    entity_type = None
    entity_box = None

    for symbol in symbols:
        symbol_id, symbol_box, type_,conf_score = symbol
        if check_text_symbol_connection(text_box, symbol_box):
            return symbol_id, 'symbol', symbol_box
        dist = bbox_min_distance(text_box, symbol_box)
        if dist < closest_distance:
            closest_distance = dist
            closest_entity = symbol_id
            entity_type = 'symbol'
            entity_box = symbol_box
    for line in lines:
        line_id, line_coords, _, _ = line
        line_box = [min(line_coords[0], line_coords[2]), min(line_coords[1], line_coords[3]), max(line_coords[0], line_coords[2]), max(line_coords[1], line_coords[3])]
        distance_to_line = point_to_rectangle_distance((np.mean([text_box[0], text_box[2]]), np.mean([text_box[1], text_box[3]])), line_box)
        if distance_to_line < closest_distance:
            closest_distance = distance_to_line
            closest_entity = line_id
            entity_type = 'line'
            entity_box = line_box

    return closest_entity, entity_type, entity_box

# Function to verify that each symbol is connected to at least one line
def verify_symbol_connections(symbols, associated_lines_symbols):
    connected_symbols = set(assoc[1] for assoc in associated_lines_symbols)
    for symbol in symbols:
        symbol_id = symbol[0]
        if symbol_id not in connected_symbols:
            raise ValueError(f"Symbol {symbol_id} is not connected to any line.")

# Function to verify that each line is connected to at least one other line
def verify_line_connections(lines, connected_lines):
    connected_lines_set = set()
    for conn in connected_lines:
        connected_lines_set.update([conn[0], conn[1]])
    for line in lines:
        line_id = line[0]
        if line_id not in connected_lines_set:
            print(f"Line {line_id} is not connected to another line.")

def draw_bezier_curve(image, start_point, control_point1, control_point2, end_point, color, thickness):
    curve_points = []
    for t in np.linspace(0, 1, 100):
        x = (1 - t)**3 * start_point[0] + 3 * (1 - t)**2 * t * control_point1[0] + 3 * (1 - t) * t**2 * control_point2[0] + t**3 * end_point[0]
        y = (1 - t)**3 * start_point[1] + 3 * (1 - t)**2 * t * control_point1[1] + 3 * (1 - t) * t**2 * control_point2[1] + t**3 * end_point[1]
        curve_points.append((int(x), int(y)))
    for i in range(len(curve_points) - 1):
        cv2.line(image, curve_points[i], curve_points[i + 1], color, thickness)

def process_image(image_path, lines_path, symbols_path, words_path):
    # Load data
    lines = np.load(lines_path, allow_pickle=True)
    symbols = np.load(symbols_path, allow_pickle=True)
    words = np.load(words_path, allow_pickle=True)

    connected_lines = []
    associated_lines_symbols = []
    thres=10
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if check_line_adjacency(lines[i][1], lines[j][1], thres*3):
                connected_lines.append((lines[i][0], lines[j][0]))

    for line in lines:
        for symbol in symbols:
            if check_line_symbol_association(line[1], symbol[1], thres/5):
                associated_lines_symbols.append((line[0], symbol[0]))
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_connected_lines_image_path = f"./output/image/{base_name}_connected_lines.jpg"
    output_line_symbol_image_path = f"./output/image/{base_name}_line_symbol.jpg"
    output_text_conn_image_path = f"./output/image/{base_name}_text_conn.jpg"
    output_npy_path = f"./output/npy/{base_name}_connections.npy"

    # Load the image
    img = cv2.imread(image_path)
    for line in lines:
        line_coords = line[1]
        cv2.line(img, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), (255, 0, 0), 2)  # Blue color (BGR)

    # Draw connections between lines
    for conn in connected_lines:
        line1 = lines[int(conn[0].split('_')[1]) - 1][1]
        line2 = lines[int(conn[1].split('_')[1]) - 1][1]
        mid1 = [(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2]
        mid2 = [(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2]
        ctrl1 = [mid1[0] + 50, mid1[1] + 50]
        ctrl2 = [mid2[0] + 50, mid2[1] + 50]
        draw_bezier_curve(img, mid1, ctrl1, ctrl2, mid2, (0, 0, 255), 2)
    cv2.imwrite(output_connected_lines_image_path, img)
    img = cv2.imread(image_path)
    for line in lines:
        line_coords = line[1]
        cv2.line(img, (line_coords[0], line_coords[1]), (line_coords[2], line_coords[3]), (255, 0, 0), 2)  # Blue color (BGR)
    
    # Draw associations between lines and symbols
    for assoc in associated_lines_symbols:
        line = lines[int(assoc[0].split('_')[1]) - 1][1]
        symbol = symbols[int(assoc[1].split('_')[1]) - 1][1]
        line_mid = [(line[0] + line[2]) / 2, (line[1] + line[3]) / 2]
        symbol_mid = [(symbol[0] + symbol[2]) / 2, (symbol[1] + symbol[3]) / 2]
        ctrl1 = [(line_mid[0] + symbol_mid[0]) / 2 + 50, (line_mid[1] + symbol_mid[1]) / 2 + 50]
        draw_bezier_curve(img, line_mid, ctrl1, ctrl1, symbol_mid, (0, 0, 255), 2)
    cv2.imwrite(output_line_symbol_image_path, img)

    # Draw text connections
    img = cv2.imread(image_path)
    text_connections = []
    for text in words:
        text_id, text_box, text_content, text_orientation = text
        x0, y0, x1, y1 = map(int, map(round, text_box))
        if (400 <= x0 <= 5500 and 300 <= y0 <= 4320) or (400 <= x1 <= 5500 and 300 <= y1 <= 4320):
            connected_entity, entity_type, entity_box = connect_text_to_closest_entity(text_box, symbols, lines)
            if not (connected_entity or entity_type or entity_box):
                raise ValueError(f"Text {text_id} is not connected to any entity.")

            text_rect = [int(text_box[0]), int(text_box[1]), int(text_box[2] - text_box[0]), int(text_box[3] - text_box[1])]
            cv2.rectangle(img, (text_rect[0], text_rect[1]), (text_rect[0] + text_rect[2], text_rect[1] + text_rect[3]), (0, 255, 0), 2)
            text_center = (int((text_box[0] + text_box[2]) / 2), int((text_box[1] + text_box[3]) / 2))
            #print(text_center)
            cv2.putText(img, text_content, text_center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

            entity_center = (int((entity_box[0] + entity_box[2]) / 2), int((entity_box[1] + entity_box[3]) / 2))
            cv2.line(img, text_center, entity_center, (0, 0, 255), 2, cv2.LINE_AA)
            text_connections.append((text_id,text_box, connected_entity, entity_type,entity_box))


    cv2.imwrite(output_text_conn_image_path, img)

    verify_symbol_connections(symbols, associated_lines_symbols)
    verify_line_connections(lines, connected_lines)

    connections = {
        'symbol_line_connections': associated_lines_symbols,
        'line_line_connections': connected_lines,
        'text_entity_connections': text_connections
    }

    np.save(output_npy_path, connections)
# Example usage:
#process_image("./0.jpg", "./output/npy/0_lines.npy", "./output/npy/0_symbols.npy", "./output/npy/0_words.npy")
