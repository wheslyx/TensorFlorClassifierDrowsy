#!/bin/sh
# Description: This function receives frozen image from training last checkpoint and detects faces in frame taken in webcam or first order priority video source.
# The output is picture stored in face_detection/output where the faces have been detected with a confidence of detection. It stores the pictures starting from 1 to infinite in png format. It uses the library image utils for video streaming
#  video_detection.py
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2

from io import StringIO
from PIL import Image
from matplotlib import pyplot as plt
from collections import defaultdict
from utils import visualization_utils as vis_util
from utils import label_map_util
import argparse  # Define the video stream

#from imutils.video import VideoStream
import imutils
import time

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()

parser.add_argument('--labels', help='Directorio a label_map.pbtxt', default ='configuracion/label_map.pbtxt')
parser.add_argument('--images', help='Directorio de las imagenes a procesar', default = 'img_pruebas')
parser.add_argument('--model', help='Directorio al modelo congelado', default = 'modelo_congelado')
args = parser.parse_args()

MAX_NUMBER_OF_BOXES = 30
MINIMUM_CONFIDENCE = 0.4
counter = 1
PATH_TO_LABELS = args.labels
PATH_TO_TEST_IMAGES_DIR = args.images
#print(PATH_TEST_IMAGE_PATHS)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize, use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)
MODEL_NAME = args.model
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# Function for giving frame proper input format to Tensor Flow
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# Video Session Start
print("[INFO] Starting video stream")
cap = cv2.VideoCapture(0) # Opencv video library is used
time.sleep(2.0) # allow camera sensor to warm up

# Load a (frozen) Tensorflow model into memory.
print('Loading model ...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine



# Detection stage
print("[DETECTION]")
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            
            # Read frame from camera
            ret, image_np = cap.read() # Start camera film
            frame = imutils.resize(image_np, width=400) #each frame is resized to a width of 300 pixels, height 300 pixels
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            frame_expanded = np.expand_dims(frame, axis=0)
            # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: frame_expanded}) # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(frame, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), CATEGORY_INDEX, use_normalized_coordinates=True, line_thickness=8) 
            # arguments are image to be modified, box coordinates where Region of Interest (ROI) is located, classes ID, scores, Category index is the class dictionary taken from /configuration/class.pbtxt, use normalize coordinates True or false, max boxes to draw is seto to all found boxes if not specified, otherwise specify, min_score_thresh is set to .5, agnostic_mode=False display only scores (Detector), line_thickness: integer (default: 4).   
            #print("Number of detections with box:")
            #detections = np.squeeze(boxes).shape[0]
            #print(detections) # Number of detections # Maximum Number of boxes available for detection
            cv2.imshow('Video Streaming', frame) #cv2.resize(image_np, (800, 600)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
    

#
#  Created by Cesar Segura on 4/4/19.
#  
