import argparse
import datetime
import importlib.util
import math
import os
import time
from random import randint
from threading import Thread

import cv2
import matplotlib.pyplot as plt
import numpy as np

from sound import SoundFunc
from lidar import lidarfunc


# Define VideoStream class to handle streaming of video from webcam in separate processing thread Source - Adrian
# Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 360), framerate=32):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)  # 0  for camera
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True


# Select tracking set up
def create_tracker_by_name(tracker_type):
    if tracker_type == tracker_types[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif tracker_type == tracker_types[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif tracker_type == tracker_types[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif tracker_type == tracker_types[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif tracker_type == tracker_types[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == tracker_types[5]:
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == tracker_types[6]:
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
        print('[ERROR] Invalid selection! Available tracker: ')
        for t in tracker_types:
            print(t.lower())

    return tracker


# The distance function that checks if an object was already tracked
def distance(object1, boxesupdt):
    x1, y1, w1, h1 = object1
    cx1 = (x1 + x1 + w1) // 2
    cy1 = (y1 + y1 + h1) // 2

    for i in boxesupdt:
        x2, y2, w2, h2 = i
        cx2 = (x2 + x2 + w2) // 2
        cy2 = (y2 + y2 + h2) // 2

        # Find out if that object was detected already
        dist = math.hypot(cx1 - cx2, cy1 - cy2)

        if dist < 200:
            return True
    return False

# Float round up function
def round_up(number, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(number * multiplier) / multiplier


# Make a figure for the LiDAR polar graph
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
ax.set_title('LiDAR scan', fontsize=18)

# Define and parse input arguments for object detection
parser = argparse.ArgumentParser()
# parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
# required=True)
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='coco')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.6)
parser.add_argument('--resolution',
                    help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x360')  # 1280x720
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Define and parse input arguments for tracker
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="KCF", help="OpenCV object tracker type")
args2 = vars(ap.parse_args())

print('[INFO] selected tracker: ' + str(args2["tracker"].upper()))

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter

    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter

    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# A fix for first label is '???', which has to be removed.
if labels[0] == '???':
    del (labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname):  # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:  # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)

# Create the Multi Tracker
multi_tracker = cv2.legacy.MultiTracker_create()

# for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

boxesupdt = []
bboxes = []
colours = []
scores_tracked = []
classes_tracked = []
distances_tracked = []

RTC = datetime.datetime.now()
RTC2 = datetime.datetime.now()

while True:
    lidar_data = lidarfunc(10)
    if ('line' in locals()):
        line.remove()
    line = ax.scatter(lidar_data.angles, lidar_data.distances, c="pink", s=5)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    plt.pause(0.01)

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Check if it's been 10 secs since the last detection
    delta = (datetime.datetime.now() - RTC)
    delta = int(delta.total_seconds())

    if delta > 10:

        RTC = datetime.datetime.now()

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Bounding box coordinates of detected objects
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]  # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]  # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                # Finding the centers of the bounding box
                xcenter1 = (boxes[i][1] + boxes[i][3]) / 2  # from the left (normalised)
                ycenter1 = (boxes[i][0] + boxes[i][2]) / 2  # from the top (normalised)
                xcenter = int((xmin + xmax) / 2)  # from the left (in pixles)
                ycenter = int((ymin + ymax) / 2)  # from the top (in pixles)

                # Finding the object distance
                angles_norm = lidar_data.angles_norm
                object_distance = round((lidar_data.distances_norm[min(range(len(angles_norm)), key=lambda i: abs(angles_norm[i] - xcenter1))]) / 100, 1)

                # selecte ROIs (ROI stands for Region of Interest)
                # Define an initial bounding box
                bbox = (xmin, ymin, abs(xmax - xmin), abs(ymax - ymin))
                # Add ROIs to list of bounding boxes
                if not distance(bbox, boxesupdt):
                    multi_tracker.add(create_tracker_by_name(args2["tracker"].upper()), frame, bbox)  # Add ROI's to tracker
                    colours.append((randint(0, 255), randint(0, 255), randint(0, 255)))  # Create random colour for each box
                    # Add ROI tracked scores and classes to new lists
                    scores_tracked.append(scores[i])
                    classes_tracked.append(classes[i])
                    distances_tracked.append(object_distance)

                    # Play sound to the user
                    SoundFunc(xcenter1, ycenter1, labels[int(classes[i])], object_distance)

    # Check if it's been 30 secs since the last voice command
    delta2 = (datetime.datetime.now() - RTC2)
    delta2 = int(delta2.total_seconds())
    if delta2 > 15:
        RTC2 = datetime.datetime.now()
        for i, distancex in enumerate(distances_tracked):
            if distancex < 3:
                (x, y, w, h) = boxesupdt[i]
                SoundFunc((x + w / 2) / imW, (y - h / 2) / imH, labels[int(classes[i])], distancex)

    boxesupdt1 = boxesupdt
    ok, boxesupdt = multi_tracker.update(frame)

    indexes1 = []
    if not ok and not np.array_equal(boxesupdt1, boxesupdt):
        #         boxesupdt_indx = [i for i, x in enumerate(boxesupdt) if (x == [0., 0., 0., 0.]).all()]
        #         boxesupdt = np.delete(boxesupdt, boxesupdt_indx, 0)
        cv2.putText(frame, 'Track Loss', (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

    # use coordinates to draw rectangle
    if len(boxesupdt) != 0:
        angles_norm = lidar_data.angles_norm
        rounded_angles_norm = [round(item, 2) for item in angles_norm]
        for i, new_box in enumerate(boxesupdt):
            (x, y, w, h) = [int(v) for v in new_box]
            if [x, y] != [0, 0] or [w, h] != [0, 0]:
                distances_tracked[i] = round((lidar_data.distances_norm[min(range(len(angles_norm)), key=lambda j: abs(angles_norm[j] - ((x + (w / 2)) / imW)))]) / 100, 1)
                object_name = labels[int(classes_tracked[i])]  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores_tracked[i] * 100))  # Example: 'person: 72%'
                label2_dist = '%s Meters' % (distances_tracked[i])  # Example: 'person: 72% 4M'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)  # Get font size
                label_ymin = max(y + h, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (x, y), (x + w, y + h), (colours[i]), 3)  # Make rectangle at the object location
                cv2.rectangle(frame, (x, label_ymin - labelSize[1] - 30), (x + labelSize[0] + 10, label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label2_dist, (x, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)  # Draw Distance label text
                cv2.putText(frame, label, (x, label_ymin - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)  # Draw label text

    # Draw framerate in corner of frame
    cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2,
                cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector and tracker', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
