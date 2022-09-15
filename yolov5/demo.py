#!/usr/bin/env python

from asyncio import protocols
import asyncio
from email.mime import image
import numpy as np
from models.experimental import attempt_load
from utils.general import imread, non_max_suppression, scale_coords
from utils.torch_utils import select_device
import cv2
import sys
import onnxruntime
import argparse
import multiprocessing
import time
import websockets

import torch


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


class Detector():

    def __init__(self):
        super(Detector, self).__init__()
        self.img_size = 640
        self.threshold = 0.6
        self.iou_thres = 0.3
        self.stride = 1
        self.weights = "D:/yolo/yolov5/runs/train/robo4_epoch150_s/weights/best.onnx"
        self.init_model()

    def init_model(self):
       # print(type(self.weights))
        sess = onnxruntime.InferenceSession(self.weights, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = sess.get_inputs()[0].name
        output_names = []
        for i in range(len(sess.get_outputs())):
           # print('output shape:', sess.get_outputs()[i].name)
            output_names.append(sess.get_outputs()[i].name)

        self.output_name = sess.get_outputs()[0].name
       # print('input name:%s, output name:%s' %
        #   (self.input_name, self.output_name))
        input_shape = sess.get_inputs()[0].shape
       # print('input_shape:', input_shape)
        self.m = sess

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img).astype(np.float16)
        img /= 255.0  # 图像归一化
        img = np.expand_dims(img, axis=0)
        assert len(img.shape) == 4

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)
        W, H = img.shape[2:]

        pred = self.m.run(None, {self.input_name: img})[0]
        pred = pred.astype(np.float32)
        pred = np.squeeze(pred, axis=0)

        boxes = []
        classIds = []
        confidences = []
        for detection in pred:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID] * detection[4]

            if confidence > self.threshold:
                box = detection[0:4]
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                classIds.append(classID)
                confidences.append(float(confidence))

        idxs = cv2.dnn.NMSBoxes(
            boxes, confidences, self.threshold, self.iou_thres)

        pred_boxes = []
        pred_confes = []
        pred_classes = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                confidence = confidences[i]
                if confidence >= self.threshold:
                    pred_boxes.append(boxes[i])
                    pred_confes.append(confidence)
                    pred_classes.append('good' if classIds[i] == 1 else 'bad')

        return im, pred_boxes, pred_confes, pred_classes


async def process(websocket, path):
    async for image in websocket:
        # nparr = np.frombuffer(img, np.uint8)
        # image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # image = cv2.imread("D:\imgData\images\maksssksksss3.png")
        
        image = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        time_start = time.time()
        shape = (det.img_size, det.img_size)
        im0, pred_boxes, pred_confes, pred_classes = det.detect(image)
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        if len(pred_boxes) > 0:
            for i, _ in enumerate(pred_boxes):
                box = pred_boxes[i]
                left, top, width, height = box[0], box[1], box[2], box[3]
                box = (left, top, left + width, top + height)
                box = np.squeeze(
                    scale_coords(shape, np.expand_dims(box, axis=0).astype(
                        "float"), im0.shape[:2]).round(),
                    axis=0).astype("int")
                x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
                cv2.rectangle(image, (x0, y0), (x1, y1),
                              (0, 0, 255), thickness=2)

                cv2.putText(image, '{0}:{1:.2f}'.format(pred_classes[i], pred_confes[i]), (x0, y0 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
                # cv2.putText(image, '{0}'.format(pred_classes[i]), (x0, y0 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)
        success, encoded_image = cv2.imencode(".png", image)
        await websocket.send(encoded_image.tobytes())
        print("send")

    # cv2.imwrite("res.png",image)


# 线程之间是共用同一片地址空间的，而进程之间所使用的是不同的内存空间，所以线程之间可以共享全局变量，而不同进程使用不同的空间，所以使用的资源本质上是不同的，所以一片空间上的变量变化了不会影响另一个空间的资源变化。
#   故若想让多进程间共同操作一个变量，只能通过创建进程时将变量作为参数传入
p = None
i = 0
det = Detector()
# python export.py --weights D:\yolo\yolov5\runs\train\robo4_epoch150_s\weights\best.pt --device 0 --half --inplace --include onnx

# def main():
#     global det
#     det = Detector()
#     process(image)


# async def echo(websocket, path):
#     async for message in websocket:
#         message = "I got your message: {}".format(message)
#         await websocket.send(message)
if __name__ == "__main__":
    # image = cv2.imread("D:\imgData\images\maksssksksss3.png")
    # main()
    # for i in range(100):
    #     process(image)

    # det = Detector()
    # for i in range(10):
    #     process(1)
    asyncio.get_event_loop().run_until_complete(websockets.serve(process, '192.168.246.15', 8765))
    asyncio.get_event_loop().run_forever()
