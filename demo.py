import sys
sys.path.append('model')

import cv2
import dlib
from model.xception import xception
from model.srffd import facesr, get_FaceSR_opt
from model.facesr.models.SRGAN_model import SRGANModel
import torch
from torchvision import transforms as T
import torch.nn.functional as F
import streamlit as st
import numpy as np
import tempfile

facedetector = None
device = None
model = None
sr_model = SRGANModel(get_FaceSR_opt(), is_train=False)
sr_model.load()
label_map = ['fake', 'real']
fake_color = (0,0,255)
real_color = (0,255,0)
fake_ths = 0.1

def init(modelname):
    global facedetector, device, model
    facedetector = dlib.get_frontal_face_detector()
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    axcep = xception(num_classes=2, pretrained='').to(device)
    if modelname == 'xception':
        axcep.load_state_dict(torch.load('weights/xception0_epoch10.pth', map_location=torch.device('cpu')))
        model = axcep
    if modelname == 'Axception' or modelname == 'SRFFD':
        axcep.load_state_dict(torch.load('weights/xception1_epoch10.pth', map_location=torch.device('cpu')))
        model = axcep
    model.eval()

def getfaceareas(img, expansion_factor = 0.2):
    """
    返回areas, 元素为(lf_x, lf_y, width, height)
    """
    areas = []
    faces = facedetector(img)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        # 扩大矩形框，例如增加20%的边界
        new_x = max(0, x - int(w * expansion_factor))
        new_y = max(0, y - int(h * expansion_factor * 2))
        new_w = min(img.shape[1], w + int(w * expansion_factor * 2))
        new_h = min(img.shape[0], h + int(h * expansion_factor * 2))
        areas.append((new_x, new_y, new_w, new_h))
    return areas

def ffd_image(img, sr=False):
    areas = getfaceareas(img)
    if len(areas) > 0:
        # 在帧上运行人脸检测器，获取人脸区域坐标
        x,y,w,h = areas[0]
        # 针对每个检测到的人脸区域进行模型推理
        face_img = img[y:y+h, x:x+w]
        if sr:
            face_img = facesr(img, sr_model)
        face_img = T.ToTensor()(face_img).unsqueeze(0).to(device)
        prediction = F.softmax(model(face_img), dim=1)
        label_idx = torch.argmax(prediction,1).item()
        label = label_map[label_idx]
        confidence = round(prediction[0][label_idx].item(), 4)
        # 在帧上标注人脸区域和预测结果
        color =  real_color if label=='real' else fake_color
        cv2.rectangle(img, (x, y), (x + w, y + h), color , 6)
        cv2.putText(img, label, (x, y+h+36), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        return img, label, confidence, face_img.permute(0,2,3,1).contiguous().numpy()
    else:
        return img, None, None, None

def ffd_video(path, fps):
    fakearr = []
    # 使用 OpenCV 打开视频流
    video_capture = cv2.VideoCapture(path)
    # 获取视频帧的宽度和高度
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 设置视频保存路径及编解码器
    output_path = f'results/result.avi'
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 使用 H.264 编码器

    # 打开视频写入器
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    # 设定处理的时间
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))
    # 获得视频总帧数
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获得视频总时长（秒）
    total_seconds = total_frames / video_capture.get(cv2.CAP_PROP_FPS)
    period = int(total_frames / (total_seconds * fps))
    # print(total_frames, total_seconds,fps, period)
    frame_count = 0  # 帧数计数器
    detect_num = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_count % period != 0: 
            frame_count += 1
            continue
        # 在帧上运行人脸检测器，获取人脸区域坐标
        areas = getfaceareas(frame)
        if len(areas) > 0:
            x,y,w,h = areas[0]

            # 针对每个检测到的人脸区域进行模型推理
            face_img = frame[y:y+h, x:x+w]
            face_img = T.ToTensor()(face_img).unsqueeze(0).to(device)
            prediction = F.softmax(model(face_img), dim=1)
            label_idx = torch.argmax(prediction,1).item()
            label = label_map[label_idx]
            confidence = round(prediction[0][label_idx].item(), 4)

            # 在帧上标注人脸区域和预测结果
            color =  real_color if label=='real' else fake_color
            cv2.rectangle(frame, (x, y), (x + w, y + h), color , 6)
            cv2.putText(frame, label, (x, y+h+36), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
            if label == 'fake':
                fakearr.append((label, confidence, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            # 保存
            out.write(frame)
            detect_num += 1
        frame_count += 1

    # 释放资源并关闭窗口
    video_capture.release()
    out.release()
    return output_path, fakearr, detect_num


def generate():
    # body
    st.header('伪脸检测Demo演示')
    st.caption('Author: 黄正超、牟准')
    st.caption('清华大学深圳国际研究生院视觉信息处理实验室')
    # sidebar
    scene = st.sidebar.selectbox('选择应用场景', ('图像级检测', '视频级检测', '实时检测'))
    modelname = st.sidebar.selectbox('模型选择', ('xception', 'Axception', 'SRFFD'), index=1, help="xception: 基线模型；\n Axception: 适用于真实世界场景；\n SRFFD: 适用于真实世界低分辨率场景")
    init(modelname)
    # 图像级检测
    if scene == '图像级检测':
        uploaded_file = st.file_uploader("请上传待检测图像文件", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            # 显示上传的图像
            imgdata = uploaded_file.read()
            img = np.frombuffer(imgdata, np.uint8) 
            # 通过 OpenCV 读取图像数据为 RGB 矩阵
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 显示上传的图像
            st.image(img, channels="RGB")
            # 开始检测
            if st.button("开始检测"):
                # 显示加载状态
                with st.spinner("正在检测，请稍候..."):
                    resimg, label, confidence, faceimg = ffd_image(img, modelname=='SRFFD')
                    # 显示检测结果
                    if modelname == 'SRFFD':
                        st.image(faceimg, clamp=True,channels='RGB')
                    st.image(resimg, channels='RGB')
                    st.success('检测成功，结果：{}，置信度：{}'.format(label, confidence))
    # 视频级检测
    if scene == '视频级检测':
        uploaded_file = st.file_uploader("请上传待检测上传视频文件", type=["mp4", "avi", "mov", "wmv"])
        if uploaded_file is not None:
            # 显示上传的视频
            videodata = uploaded_file.read()
            video = np.frombuffer(videodata, np.uint8)
            # 将视频数据写入临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(video)
                temp_file_path = temp_file.name
            cap = cv2.VideoCapture(temp_file_path)
            # 获得视频帧率
            fps = cap.get(cv2.CAP_PROP_FPS)
            # st.video(video)
            target_fps = st.sidebar.slider('请选择视频帧率', min_value=int(1), max_value=int(fps), value=int(fps/2), help='为提升检测效率，可适当降低视频帧率')
            # 开始检测
            if st.button("开始检测"):
                # 显示加载状态
                with st.spinner("正在检测，请稍候..."):
                    outputpath, fakearr, frames = ffd_video(temp_file_path, target_fps)
                    # 显示检测结果
                    resvideo = open(outputpath, 'rb')
                    st.markdown('### 检测结果视频')
                    st.video(resvideo)
                    fakenum = len(fakearr)
                    if fakenum > 0:
                        st.markdown('### 视频中检测到的异常序列')
                        st.success('一共检测到{}个异常片段'.format(len(fakearr)))
                        num_images = fakenum
                        # 定义每行最多可以容纳的图片数量
                        max_images_per_row = 5
                        # 计算行数
                        num_rows = (num_images + max_images_per_row - 1) // max_images_per_row
                        # 循环遍历每一行
                        for i in range(num_rows):
                            # 创建一个水平布局，每个布局最多包含 5 列
                            columns = st.columns(max_images_per_row)
                            # 计算当前行的起始索引和结束索引
                            start_index = i * max_images_per_row
                            end_index = min((i + 1) * max_images_per_row, num_images)
                            # 在当前行的每一列显示图片
                            for j, column in enumerate(columns):
                                if start_index + j < end_index:
                                    # 显示图片
                                    column.image(fakearr[start_index + j][2], use_column_width=True)
                    else:
                        st.success('未检测到异常片段')
                    if fakenum / frames > fake_ths:
                        st.markdown("<font color='red'>异常片段较多，疑似采用AI换脸技术，请谨慎识别！</font>", unsafe_allow_html=True)
    # 实时监测
    if scene == '实时检测':
        st.markdown('摄像头实时视频')
        # 创建一个空容器用于显示视频
        video_container = st.empty()
        open_btn = st.button('打开摄像头')
        if open_btn:
            video_container = st.empty()
            stop_btn = st.button('关闭摄像头')
            # 打开摄像头
            cap = cv2.VideoCapture(0)
            # 设置视频流的宽度和高度
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # 在页面上显示实时视频
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 将 OpenCV 图像转换为 RGB 格式以在 Streamlit 中显示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 清空容器并显示新的视频帧
                video_container.image(ffd_image(frame_rgb)[0], channels="RGB")
                # 检查用户是否点击了停止按钮
                if stop_btn:
                    break
            # 释放摄像头资源
            cap.release()

if __name__ == "__main__":
    generate()