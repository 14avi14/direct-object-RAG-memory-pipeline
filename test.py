"""test.py - a file simply meant for testing things

**Ensure that `run_pipeline.py` is RUNNING beforehand in a different terminal**
Run the file, and enter either a video file name, e.g. "video.mp4", or
click ENTER to use webcam footage. This will simulate the car sending
footage over to the server. The car location data is fake and static, and
so is the depth data."""

from ultralytics import YOLO, FastSAM
import cv2
import numpy as np
import socket
import time
import json
import os

import socket_ops

file_name = input("Input the chosen video file, or click ENTER for webcam: ")

if os.path.isfile(file_name):
    cap = cv2.VideoCapture(file_name)
else:
    print("No video file, using webcam.")
    cap = cv2.VideoCapture(0)

conn = socket.socket()

conn.connect(socket_ops.SERVER_ADDRESS)

if not cap.isOpened():
    print("Cannot access video, exiting.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    depth_frame = np.ones(frame.shape[:2]) * 255 # Default to 255mm away everywhere(FAKE)

    # FAKE location data
    default_car_state = {
        "heading_deg": 0,
        "position_xy": [0, 0]
    }

    # Display the resulting frame
    cv2.imshow('frame', frame)

    
    socket_ops.send_image(conn, frame)
    socket_ops.send_image(conn, depth_frame)
    socket_ops.send_string(conn, json.dumps(default_car_state))
    socket_ops.send_string(conn, str(time.time()))

    result = json.loads(socket_ops.recieve_string(conn))
    if cv2.waitKey(5) == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
