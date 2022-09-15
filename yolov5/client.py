import asyncio
import websockets
import cv2
import numpy as np

image = cv2.imread("D:\imgData\images\maksssksksss3.png")
success, encoded_image = cv2.imencode(".png", image)

async def hello(uri):
    async with websockets.connect(uri) as websocket:
        while(True):
            await websocket.send(encoded_image.tobytes())
            recv_text = await websocket.recv()
            img=np.frombuffer(recv_text,np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            cv2.imwrite('out1.jpg', img)
            print(len(recv_text))
# 多个协程是在同一个线程中执行的，因此，await一个协程的时候，其他协程可以可以在当前线程中继续执行。
# 如果是在线程中同步执行，那么一个任务阻塞的时候，整个线程都阻塞了，没办法做其他任务

asyncio.get_event_loop().run_until_complete(
    hello('ws://192.168.246.15:8765'))

