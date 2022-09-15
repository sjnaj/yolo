'''
Author: fengsc
Date: 2022-09-12 18:58:49
LastEditTime: 2022-09-13 00:38:52
'''
import asyncio
import websockets
import cv2
import numpy as np
from sympy import im
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
import time

def process():
    # weights = 'yolov5/runs/train/robo4_epoch150_s/weights/best.pt'
    weights="D:\\yolo\\yolov5\\runs\\train\\robo4_epoch150_s\\weights\\best.torchscript"
    w = str(weights[0] if isinstance(weights, list) else weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(
        weights, device)  # 加载模型
    height, width = 640, 640
    img0 = cv2.imread('D:\imgData\images\maksssksksss3.png')
    img = cv2.resize(img0, (height, width))  # 尺寸变换
    img = img / 255.
    img = img[:, :, ::-1].transpose((2, 0, 1))  # HWC转CHW
    img = np.expand_dims(img, axis=0)  # 扩展维度至[1,3,640,640]
    img = torch.from_numpy(img.copy())  # numpy转tensor
    img = img.to(torch.float32)  # float64转换float32
    t1=time.time()
    # pred = model(img, augment='store_true', visualize='store_true')[0]
    
    pred=model(img.cuda())[0].cpu()
    
    t2=time.time()
    print(t2-t1)
    
    # pred.clone().detach()
    pred = non_max_suppression(
        pred, 0.25, 0.45, None, False, max_det=1000)  # 非极大值抑制

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                # 输出结果：xyxy检测框左上角和右下角坐标，conf置信度，cls分类结果
                # print('{},{},{}'.format(xyxy, conf.numpy(), cls.numpy()))
                img0 = cv2.rectangle(img0, (int(xyxy[0].numpy()), int(xyxy[1].numpy())), (int(
                    xyxy[2].numpy()), int(xyxy[3].numpy())), (0, 255, 0), 2)
                cv2.putText(img0, str(i), (int(xyxy[0].numpy()), int(
                    xyxy[1].numpy())),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imwrite('out.jpg', img0)

async def echo(websocket, path):
    async for message in websocket:
        message = "I got your message: {}".format(message)
        await websocket.send(message)
asyncio.get_event_loop().run_until_complete(websockets.serve(echo, 'localhost', 8765))
asyncio.get_event_loop().run_forever()
if __name__ == '__main__':
    for i in range(0,100):
        process();