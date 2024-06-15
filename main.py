from ultralytics import YOLO, RTDETR
from mmocr.apis import MMOCRInferencer
from deeplsd.models.deeplsd_inference import DeepLSD
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from torchvision.ops import nms
import cv2
import argparse
import time
from utils.symbol import symbol_detection
from utils.ocr import ocr_infer
from utils.remove import remove_boxes
from utils.lines import detect_and_merge_lines
from utils.connect import process_image

seed=114154
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
randomness = dict(seed=seed)


ckpt = 'weights/deeplsd_md.tar'
ckpt = torch.load(str(ckpt), map_location='cpu')
conf = {
    'sharpen': True,  # Use the DF normalization (should be True)
    'detect_lines': True,  # Whether to detect lines or only DF/AF
    'line_detection_params': {
        'merge': True,  # Whether to merge close-by lines
        'optimize': True,  # Whether to refine the lines after detecting them
        'use_vps': True,  # Whether to use vanishing points (VPs) in the refinement
        'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
        'grad_thresh': 4,
        'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
    }
}

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

detr_model = RTDETR("weights/train11/weights/best.pt")
yolo_model = YOLO("weights/train4/weights/best.pt")
line_model = DeepLSD(conf)
line_model.load_state_dict(ckpt['model'])
ocr_model = MMOCRInferencer(det='DBNetpp', rec='ABINet', device=device)

def main(img_path):
    start=time.time()
    symbol_npy_path=symbol_detection(detr_model, yolo_model,img_path)
    sym_t=time.time()-start
    ocr_npy_path=ocr_infer(ocr_model, img_path)
    ocr_t=time.time()-sym_t-start
    npy_path=[ocr_npy_path, symbol_npy_path]
    img=remove_boxes(img_path,npy_path)[:, :, ::-1]
    rm_t=time.time()-ocr_t-start
    #cv2.imwrite('0_remove.jpg',img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # 保存黑白图像
    cv2.imwrite('gray.jpg', binary_image)
    line_npy_path = detect_and_merge_lines('gray.jpg',img_path, line_model,device)
    line_t=time.time()-rm_t-start
    process_image(img_path,line_npy_path,symbol_npy_path,ocr_npy_path)
    pro_t=time.time()-line_t-start
    print("symbol time:",sym_t)
    print("ocr time:",ocr_t)
    print("remove time:",rm_t)
    print("line time:",line_t)
    print("conn time:",pro_t)
    print("whole time:",sym_t+ocr_t+rm_t+line_t+pro_t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image with YOLO, RTDETR, MMOCR, and DeepLSD models")
    parser.add_argument('--img_path', type=str, required=True,default='0.jpg', help="Path to the input image")
    #parser.add_argument('--device', type=str, default='cuda', help="Device to run the models on (e.g., 'cuda' or 'cpu')")
    args = parser.parse_args()
    main(args.img_path)