from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True,
                    help='path to input video file')
args = parser.parse_args()

# open video stream
stream = cv2.VideoCapture(args.video)
fps = FPS().start()

while True:
    (grabbed, frame) = stream.read()
    
    if not grabbed:
        break
    frame = imutils.resize(frame, width=450)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = np.dstack([frame, frame, frame])
 
    cv2.putText(frame, "Slow Method", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    fps.update()
    fps.stop()
    print(fps.fps())
 
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()
    