import myUtils, configRead, opcServer
import cv2
import tensorflow
import keras
import numpy as np
import pickle
import sys
from configparser import ConfigParser
import csv

print('Starting up...')
# APP CONFIGURATION LOAD
parser = ConfigParser()
parser.read('config.ini')

# TODO
# Load config for all objects and return object list
show_box_images = False

# NEURAL NETWORK SETTINGS ###
# NEURAL NETWORK DETECTION THRESHOLD ###
# cnnCertainty = 0.9  # LEVEL OF NEURAL NETWORK CERTAINTY 0-1 (0-100%) HOW CONFIDENT IS THE CNN IN THE RESULT
cnnCertainty = parser.getfloat('neural_network', 'mincertainty')

# CROPPING AND SCALING
# scaleFactor = 10
scaleFactor = parser.getint('image_processing', 'scalefactor', fallback=10)

# CROPPING COORDINATES FOR FIXED BOXES
x1 = parser.getint('box_coordinates', 'x1')
y1 = parser.getint('box_coordinates', 'y1')
h1 = parser.getint('box_coordinates', 'h1')
w1 = parser.getint('box_coordinates', 'w1')
x2 = parser.getint('box_coordinates', 'x2')
y2 = parser.getint('box_coordinates', 'y2')
h2 = parser.getint('box_coordinates', 'h2')
w2 = parser.getint('box_coordinates', 'w2')

leftLadleBoxLocked = False
rightLadleBoxLocked = False

# thresholdladleLeft = 220
# thresholdladleRight = 180
thresholdladleleft = parser.getint('image_processing', 'thresholdladleleft')
thresholdladleright = parser.getint('image_processing', 'thresholdladleright')

# SINGLE DIGIT BOXES
# minSnglDigitBoxWidth = 100
# maxSnglDigitBoxWidth = 250
# minSnglDigitBoxHeigth = 200
# maxSnglDigitBoxHeigth = 350
minSnglDigitBoxWidth = parser.getint('single_digit_boxes', 'minsngldigitboxwidth')
maxSnglDigitBoxWidth = parser.getint('single_digit_boxes', 'maxsngldigitboxwidth')
minSnglDigitBoxHeigth = parser.getint('single_digit_boxes', 'minsngldigitboxheigth')
maxSnglDigitBoxHeigth = parser.getint('single_digit_boxes', 'maxsngldigitboxheigth')

# minDblDigitBoxWidth = 320
# maxDblDigitBoxWidth = 500
# minDblDigitBoxHeigth = 250
# maxDblDigitBoxHeigth = 450
minDblDigitBoxWidth = parser.getint('double_digit_boxes', 'mindbldigitboxwidth')
maxDblDigitBoxWidth = parser.getint('double_digit_boxes', 'maxdbldigitboxwidth')
minDblDigitBoxHeigth = parser.getint('double_digit_boxes', 'mindbldigitboxheigth')
maxDblDigitBoxHeigth = parser.getint('double_digit_boxes', 'maxdbldigitboxheigth')

# minFillDregree = 0.2
# maxFillDegree = 0.9
minFillDregree = parser.getfloat('box_filter_params', 'minfilldregree')
maxFillDegree = parser.getfloat('box_filter_params', 'maxfilldegree')

### OPTIONS ###
dataDumpEnabled = parser.getboolean('settings',
                                    'datadumpenabled')  # Stores images from video feed every 1 min and writes debug data on s CSV file
leftScanEnabled = parser.getboolean('settings', 'leftscanenabled')
rightScanEnabled = parser.getboolean('settings', 'rightscanenabled')
colorFilterON = parser.getboolean('settings', 'colorfilteron')
usePresetHSVFilter = parser.getboolean('settings', 'usepresethsvfilter')

### COLOR HSF FILTER PRESET (HueMin,HueMax,SatMin,SatMax,ValMin,ValMax)

colorHSVFilter1 = (parser.getint('HSV_Filter', 'huemin1'),
                   parser.getint('HSV_Filter', 'huemax1'),
                   parser.getint('HSV_Filter', 'satmin1'),
                   parser.getint('HSV_Filter', 'satmax1'),
                   parser.getint('HSV_Filter', 'valmin1'),
                   parser.getint('HSV_Filter', 'valmax1'))  # NUMBER COLOR PRESET HSV RANGE

colorHSVFilter2 = (parser.getint('HSV_Filter', 'huemin2'),
                   parser.getint('HSV_Filter', 'huemax2'),
                   parser.getint('HSV_Filter', 'satmin2'),
                   parser.getint('HSV_Filter', 'satmax2'),
                   parser.getint('HSV_Filter', 'valmin2'),
                   parser.getint('HSV_Filter', 'valmax2'))  # NUMBER COLOR PRESET HSV RANGE

# maxNumberSamples = 50  # SAMPLES TO TAKE BEFORE DECIDING THE LADLE NUMBER IS CORRECT, ASK MANY ANSWER ONCE
maxNumberSamples = parser.getint('neural_network',
                                 'validationsamples')  # SAMPLES TO TAKE BEFORE DECIDING THE LADLE NUMBER IS CORRECT, ASK MANY ANSWER ONCE

### CREATE AND INITIALISE HSV TRACKBARS

myUtils.createHSVTrackbars("HSV Left Filter", HueMin=colorHSVFilter1[0], HueMax=colorHSVFilter1[1],
                           SatMin=colorHSVFilter1[2],
                           SatMax=colorHSVFilter1[3], ValMin=colorHSVFilter1[4], ValMax=colorHSVFilter1[5])

myUtils.createHSVTrackbars("HSV Right Filter", HueMin=colorHSVFilter2[0], HueMax=colorHSVFilter2[1],
                           SatMin=colorHSVFilter2[2],
                           SatMax=colorHSVFilter2[3], ValMin=colorHSVFilter2[4], ValMax=colorHSVFilter2[5])

### CREATE AND INITIALISE THRESOLDING ADJUST TRACKBARS

myUtils.createThresTrackbars("ThresholdTrackBars", thresholdladleleft, thresholdladleright)

# ====== DETECTION BOXES INITIALISATION ======

detection_boxes = [myUtils.DetectionBox() for i in range(2)]

left_ladle_box = detection_boxes[0]
right_ladle_box = detection_boxes[1]

left_ladle_box.title = 'Left Ladle'
left_ladle_box.corner_color = (0, 0, 255)
left_ladle_box.title_thickness = 1
left_ladle_box.update(x1, y1)
left_ladle_box.colorHSVFilter = colorHSVFilter1
left_ladle_box.thresholdValue = thresholdladleleft
left_ladle_box.validation_sample_target = maxNumberSamples

right_ladle_box.title = 'Right Ladle'
right_ladle_box.corner_color = (255, 0, 0)
right_ladle_box.title_thickness = 1
right_ladle_box.colorHSVFilter = colorHSVFilter2
right_ladle_box.thresholdValue = thresholdladleright
right_ladle_box.validation_sample_target = maxNumberSamples

right_ladle_box.update(x2, y2)
# ============================================

# url = "rtsp://10.81.98.80/?line=4?inst=2"
# url = "rtsp://10.81.98.165/?line=1"
url = parser.get('video_feed', 'url')

cap = cv2.VideoCapture(url)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open camera feed")

ret, img = cap.read()
print(img.shape)


def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_MOUSEMOVE:
        for obj in detection_boxes:
            if obj.selected:
                obj.update(x - obj.selected_offset[0], y - obj.selected_offset[1])

    if event == cv2.EVENT_LBUTTONDOWN:
        for obj in detection_boxes:
            x1, y1 = obj.pos_xy
            w1, h1 = obj.size
            if x1 < x < x1 + w1 and y1 < y < y1 + h1:
                obj.selected = True
                obj.selected_offset = [x - obj.pos_xy[0], y - obj.pos_xy[1]]
                print(f'{obj.title} box selected')

    if event == cv2.EVENT_LBUTTONUP:
        for obj in detection_boxes:
            obj.selected = False
            myUtils.updateSaveBoxPosition(detection_boxes)
        # print(f'Mouse is at X={x},Y={y}')


### OPC SERVER ###
opcServerEnabled = parser.getboolean('opc-ua-server', 'enable_opc_server', fallback=False)
if opcServerEnabled:
    opc_vars = []
    opc_server = opcServer.Server()
    opc_endpoint = 'opc.tcp://' + parser.get('opc-ua-server', 'endpoint') + ':' + parser.get('opc-ua-server', 'port',
                                                                                             fallback=5000)
    opc_server.set_endpoint(opc_endpoint)

    ### Register NameSpace
    namespace = opc_server.register_namespace(parser.get('opc-ua-server', 'namespace', fallback='Not defined'))
    node = opc_server.get_objects_node()
    opc_obj = node.add_object(namespace, parser.get('opc-ua-server', 'group_name', fallback='Group Not defined'))
    for box in detection_boxes:
        opc_vars.append(opc_obj.add_variable(namespace, box.title, 0))

    print("Starting OPC Server...")
    opc_server.start()
    print("OPC-UA Server Online")
    print('Listening on', opc_endpoint)
    ### OPC SERVER END ###

### MAIN LOOP ###

def key_handler(key):
    global show_box_images

    if key == ord('d') or key == ord('D'):  # Show/Hide pre-processed image details
        show_box_images = not show_box_images
        for boxes in detection_boxes:
            boxes.show_processed_images = show_box_images

    if key == ord('q') or key == ord('Q'):  # Quits the application
        cap.release()
        cv2.destroyAllWindows()
        # sys.exit()
        opc_server.stop()
        quit()

while True:
    timer = cv2.getTickCount()
    top_left_text = []
    top_left_text.append('-' * 25)
    ret, img = cap.read()
    if ret == False:
        continue



    _, _, left_ladle_box.colorHSVFilter = myUtils.captureHSVTrackbarValues("HSV Left Filter")
    _, _, right_ladle_box.colorHSVFilter = myUtils.captureHSVTrackbarValues("HSV Right Filter")
    # left_ladle_box.thresholdValue, right_ladle_box.thresholdValue = myUtils.captureThresTrackbarsValues()

    for i, box in enumerate(detection_boxes):

        img = box.detect_and_draw(img)
        top_left_text.append(f'{box.title} => {box.validated_number} - Thres = {box.thresholdValue}')

        # cv2.imshow(f'{box.title} - Left digit', box.left_digit_img)
        # cv2.imshow(f'{box.title} - Right digit', box.right_digit_img)
        if opcServerEnabled: opc_vars[i].set_value(box.validated_number)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # MENU
    top_left_text.insert(0, f'FPS {int(fps)}')
    top_left_text.append('-' * 25)
    top_left_text.append('"D - Show Details')
    top_left_text.append('')
    top_left_text.append('"q" to quit application')
    img = myUtils.draw_text_on_top_left_corner(img, top_left_text)
    # MENU END
    cv2.imshow('Video Feed', img)
    cv2.setMouseCallback('Video Feed', mousePoints)

    key = cv2.waitKey(parser.getint('video_feed', 'frame_delay', fallback=500)) & 0xFF
    if key < 255: key_handler(key)
