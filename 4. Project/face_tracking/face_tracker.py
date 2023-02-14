from imutils.video import VideoStream, FPS
from centroidtracker import CentroidTracker
import numpy as np
import argparse
import imutils
import time
import cv2

prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
# model  = 'opencv_face_detector.caffemodel'
confidence = 0.8

ct = CentroidTracker()
(H, W) = (None, None)

print("[INFO] loading model....")
net = cv2.dnn.readNetFromCaffe(prototxt, model)


parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True, help='video or camera')
args = parser.parse_args()

if args.source == '0':
    print("[INFO] starting video streams...")
    stream = VideoStream(src=0).start()
else:
    print("[INFO] starting video Capture...")
    stream = cv2.VideoCapture(args.source)
    
fps = FPS().start()
time.sleep(2.0)

while True:
    try:
        if args.source == '0':
            frame = stream.read()
            frame = imutils.resize(frame, width=500)
        else:
            (grabbed, frame) = stream.read()
            frame = imutils.resize(frame, width=500)
        
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            
        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []
        
        # detections = [[]]
        for i in range(0, detections.shape[2]):
            if detections[0, 0, i, 2] > confidence:
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                rects.append(box.astype("int"))
                
                (startX, startY, endX, endY) = box.astype('int')
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
        objects = ct.update(rects)
        
        for (objectID, centroid) in objects.items():
            text = f'ID {objectID}'
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] -10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0,255,0), -1)
            
    except Exception as e:
        # print('No Objects')
        print(e)
        
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    fps.update()
    fps.stop()
    print(fps.fps())
    

cv2.destropyAllWindows()
stream.stop()   