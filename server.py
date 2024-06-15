from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import gradio as gr
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO, RTDETR
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms
import cv2
os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "tmp")
OPENAI_API_KEY = "" #use a openai api key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

detr_model = RTDETR("weights/train11/weights/best.pt")
yolo_model = YOLO("weights/train4/weights/best.pt")


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
    # 定义设备名称列表
    device_names = [
        "闸阀", "球阀", "截止阀", "通用阀", "球阀", "蝶形阀", "旋塞阀", "止回阀", "隔膜阀", "针阀",
        "半关闭的通用阀", "关闭的通用阀", "关闭的截止阀", "控制阀", "通用旋转阀", "关闭的通用旋转阀",
        "开式垫片", "关闭的带双圈的盲板", "打开的带双圈的盲板", "减径管", "法兰", "管路混合器", "换热器",
        "中间箭头", "流量计", "GRI 808", "RO-10 871", "SDL 973", "DDL 686", "STA", "ZLO 946", "121 LG-10 190"
    ]

    # 定义结构化数组的数据类型
    dtype = [('symbol', 'U10'),  ('device_name', 'U32')]
    results = np.zeros(len(cls), dtype=dtype)

    for i in range(len(cls)):
        symbol = f'{i+1}'
        bbox = xyxy[i].cpu().numpy().astype('int')
        conf_score = float(conf[i].cpu().numpy())
        device_name = device_names[int(cls[i])]
        results[i] = (symbol,device_name)
    return results
    # 保存结果
    #np.save(file_name, results)


def symbol_detection(detr_model, yolo_model, image_path,ret_type=0):
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
    ret=save_results(cls_nms, xyxy_nms, conf_nms, file_name=output_npy_path)

    # 绘制边界框并保存图片
    return ret

def process_image1(image):

    detection_results = symbol_detection(detr_model, yolo_model, image,ret_type=1)

    template = """
    You are an intelligent assistant. You are given the detection results of symbols from an P&ID image.
    The data contains symbols' serial number, bounding boxes, class IDs, and confidence scores.
    Provide a description of how many symbols and what symbols are in the image. use chinese and answer as briefly as possible.
    Data: {data}
    """
    prompt = PromptTemplate(template=template, input_variables=["data"])
    llm = OpenAI(temperature=0,max_tokens=2500)  
    chain=prompt|llm  
    #import pdb;pdb.set_trace()
    response = chain.invoke({"data": detection_results})
    return response,detection_results

# gr.Interface(
#     fn=process_image1,
#     inputs=gr.Image(type="filepath"),
#     outputs=[gr.Textbox(),gr.Textbox()],
#     title="Symbol Detection and Description",
#     description="Upload an image to detect symbols and get detailed descriptions.",
# ).launch()
def chat_with_detection(input_text, detection_results):
    prompt = f"""
    You are an intelligent assistant. You are given the detection results of symbols from an P&ID image.
    The data contains symbols' serial number and device names. Answer in Chinese.
    User's query: {input_text}
    Detection results: {detection_results}
    """
    llm = OpenAI(temperature=0, max_tokens=2500)
    response = llm(prompt)
    return response

with gr.Blocks() as demo:
    state = gr.State()
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload an image")
        text_output = gr.Textbox(label="Detection Results")
    with gr.Row():
        user_input = gr.Textbox(label="Your question")
        chat_output = gr.Textbox(label="Chat Response")
    with gr.Row():
        detect_button = gr.Button("Detect Symbols")
        chat_button = gr.Button("Ask Question")

    def detect(image):
        response, detection_results = process_image1(image)
        state.value = detection_results
        return response, str(detection_results)

    def chat(input_text):
        detection_results = state.value
        if detection_results is not None:
            response = chat_with_detection(input_text, detection_results)
            return response
        else:
            return "Please upload an image and perform detection first."

    detect_button.click(detect, inputs=image_input, outputs=[text_output, gr.State()])
    chat_button.click(chat, inputs=user_input, outputs=chat_output)

demo.launch()