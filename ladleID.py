import myUtils, opcServer
import cv2
import tensorflow
import keras
import numpy as np
import pickle
import sys
from configparser import ConfigParser
import csv



print('Starting up...')
### APP CONFIGURATION LOAD ##
parser = ConfigParser()
parser.read('config.ini')

### OPC SERVER ###
opcServerEnabled = parser.getboolean('opc-ua-server','enable_opc_server',fallback=False)
if opcServerEnabled:
    server = opcServer.Server()
    opcEndpoint = 'opc.tcp://' + parser.get('opc-ua_server','endpoint') +':'+ parser.get('opc-ua_server','port',fallback=5000)
    server.set_endpoint(opcEndpoint)

    ### Register NameSpace
    namespace = server.register_namespace("Ladles")
    node = server.get_objects_node()
    ladlesOPCObj = node.add_object(namespace, 'Ladle Numbers')
    ladleLeftOPCVar = ladlesOPCObj.add_variable(namespace, "Left Ladle No", 0)
    ladleRightOPCVar = ladlesOPCObj.add_variable(namespace, "Right Ladle No", 0)

    print("Starting OPC Server...")
    server.start()
    print("OPC Server Online")
    ### OPC SERVER END ###

### NEURAL NETWORK SETTINGS ###
### NEURAL NETWORK DETECTION THRESHOLD ###
# cnnCertainty = 0.9  # LEVEL OF NEURAL NETWORK CERTAINTY 0-1 (0-100%) HOW CONFIDENT IS THE CNN IN THE RESULT
cnnCertainty = parser.getfloat('neural_network','mincertainty')

### LOAD CONVOLUTIONAL NEURAL NETWORK MODEL
pickle_in = open("model_trained_20.p", "rb")
model = pickle.load(pickle_in)


# CROPPING AND SCALING
# scaleFactor = 10
scaleFactor = parser.getint('image_processing','scalefactor', fallback=10)

# CROPPING COORDINATES FOR FIXED BOXES

# x1, y1, w1, h1 = 100, 310, 100, 100
# x2, y2, w2, h2 = 530, 330, 100, 100
x1 = parser.getint('box_coordinates','x1')
y1 = parser.getint('box_coordinates','y1')
h1 = parser.getint('box_coordinates','h1')
w1 = parser.getint('box_coordinates','w1')
x2 = parser.getint('box_coordinates','x2')
y2 = parser.getint('box_coordinates','y2')
h2 = parser.getint('box_coordinates','h2')
w2 = parser.getint('box_coordinates','w2')

leftLadleBoxLocked = False
rightLadleBoxLocked = False

# thresholdladleLeft = 220
# thresholdladleRight = 180
thresholdladleleft = parser.getint('image_processing','thresholdladleleft')
thresholdladleright = parser.getint('image_processing','thresholdladleright')


# SINGLE DIGIT BOXES
# minSnglDigitBoxWidth = 100
# maxSnglDigitBoxWidth = 250
# minSnglDigitBoxHeigth = 200
# maxSnglDigitBoxHeigth = 350
minSnglDigitBoxWidth = parser.getint('single_digit_boxes','minsngldigitboxwidth')
maxSnglDigitBoxWidth = parser.getint('single_digit_boxes','maxsngldigitboxwidth')
minSnglDigitBoxHeigth = parser.getint('single_digit_boxes','minsngldigitboxheigth')
maxSnglDigitBoxHeigth = parser.getint('single_digit_boxes','maxsngldigitboxheigth')

# minDblDigitBoxWidth = 320
# maxDblDigitBoxWidth = 500
# minDblDigitBoxHeigth = 250
# maxDblDigitBoxHeigth = 450
minDblDigitBoxWidth = parser.getint('double_digit_boxes','mindbldigitboxwidth')
maxDblDigitBoxWidth = parser.getint('double_digit_boxes','maxdbldigitboxwidth')
minDblDigitBoxHeigth = parser.getint('double_digit_boxes','mindbldigitboxheigth')
maxDblDigitBoxHeigth = parser.getint('double_digit_boxes','maxdbldigitboxheigth')


# minFillDregree = 0.2
# maxFillDegree = 0.9
minFillDregree = parser.getfloat('box_filter_params','minfilldregree')
maxFillDegree = parser.getfloat('box_filter_params','maxfilldegree')

### OPTIONS ###
dataDumpEnabled = parser.getboolean('settings','datadumpenabled')     # Stores images from video feed every 1 min and writes debug data on s CSV file
leftScanEnabled = parser.getboolean('settings','leftscanenabled')
rightScanEnabled = parser.getboolean('settings','rightscanenabled')
colorFilterON = parser.getboolean('settings','colorfilteron')
usePresetHSVFilter = parser.getboolean('settings','usepresethsvfilter')

### COLOR HSF FILTER PRESET (HueMin,HueMax,SatMin,SatMax,ValMin,ValMax)
# colorHSVFilter1 = (0,179,0,91,33,255)  # PINK PRESET HSV RANGE
# colorHSVFilter1 = (0,179,0,72,65,176)  # NUMBER COLOR PRESET HSV RANGE

colorHSVFilter1 = (parser.getint('HSV_Filter','huemin1'),
                  parser.getint('HSV_Filter','huemax1'),
                  parser.getint('HSV_Filter','satmin1'),
                  parser.getint('HSV_Filter','satmax1'),
                  parser.getint('HSV_Filter','valmin1'),
                  parser.getint('HSV_Filter','valmax1'))  # NUMBER COLOR PRESET HSV RANGE

colorHSVFilter2 = (parser.getint('HSV_Filter','huemin2'),
                  parser.getint('HSV_Filter','huemax2'),
                  parser.getint('HSV_Filter','satmin2'),
                  parser.getint('HSV_Filter','satmax2'),
                  parser.getint('HSV_Filter','valmin2'),
                  parser.getint('HSV_Filter','valmax2'))  # NUMBER COLOR PRESET HSV RANGE

# maxNumberSamples = 50  # SAMPLES TO TAKE BEFORE DECIDING THE LADLE NUMBER IS CORRECT, ASK MANY ANSWER ONCE
maxNumberSamples = parser.getint('neural_network','validationsamples')  # SAMPLES TO TAKE BEFORE DECIDING THE LADLE NUMBER IS CORRECT, ASK MANY ANSWER ONCE


### CREATE AND INITIALISE HSV TRACKBARS

myUtils.createHSVTrackbars("HSV Left Filter",HueMin=colorHSVFilter1[0],HueMax=colorHSVFilter1[1],SatMin=colorHSVFilter1[2],
                           SatMax=colorHSVFilter1[3],ValMin=colorHSVFilter1[4],ValMax=colorHSVFilter1[5])

myUtils.createHSVTrackbars("HSV Right Filter",HueMin=colorHSVFilter2[0],HueMax=colorHSVFilter2[1],SatMin=colorHSVFilter2[2],
                           SatMax=colorHSVFilter2[3],ValMin=colorHSVFilter2[4],ValMax=colorHSVFilter2[5])


### CREATE AND INITIALISE THRESOLDING ADJUST TRACKBARS

myUtils.createThresTrackbars("ThresholdTrackBars",thresholdladleleft,thresholdladleright)

# ====== DETECTION BOXES INITIALISATION ======

detection_boxes = [myUtils.DetectionBox() for i in range(2)]

left_ladle_box = detection_boxes[0]
right_ladle_box = detection_boxes[1]

left_ladle_box.title = 'Left Ladle'
left_ladle_box.corner_color = (0,0,255)
left_ladle_box.title_thickness = 1
left_ladle_box.update(x1,y1)
left_ladle_box.colorHSVFilter = colorHSVFilter1
left_ladle_box.thresholdValue = thresholdladleleft

right_ladle_box.title = 'Right Ladle'
right_ladle_box.corner_color = (255,0,0)
right_ladle_box.title_thickness = 1
right_ladle_box.colorHSVFilter = colorHSVFilter2
right_ladle_box.thresholdValue = thresholdladleright

right_ladle_box.update(x2,y2)
# ============================================

# url = "rtsp://10.81.98.80/?line=4?inst=2"
# url = "rtsp://10.81.98.165/?line=1"
url = parser.get('video_feed','url')

cap = cv2.VideoCapture(url)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open camera feed")

ret, img = cap.read()
print(img.shape)



def mousePoints(event,x,y,flags,params):

    if event == cv2.EVENT_MOUSEMOVE:
        for obj in detection_boxes:
            if obj.picked:
                obj.update(x-obj.picked_offset[0],y-obj.picked_offset[1])

    if event == cv2.EVENT_LBUTTONDOWN:
        for obj in detection_boxes:
            x1, y1 = obj.pos_xy
            w1, h1 = obj.size
            if x1 < x < x1 + w1 and y1 < y < y1 + h1:
                obj.picked = True
                obj.picked_offset = [x-obj.pos_xy[0],y-obj.pos_xy[1]]
                print(f'Picked box {obj.title}')

    if event == cv2.EVENT_LBUTTONUP:
        for obj in detection_boxes:
            obj.picked = False
        # print(f'Mouse is at X={x},Y={y}')

### MAIN LOOP ###

while True:
    timer = cv2.getTickCount()
    ret, img = cap.read()
    if ret == False:
        continue



    _, _, left_ladle_box.colorHSVFilter = myUtils.captureHSVTrackbarValues("HSV Left Filter")
    _, _, right_ladle_box.colorHSVFilter = myUtils.captureHSVTrackbarValues("HSV Right Filter")
    left_ladle_box.thresholdValue, right_ladle_box.thresholdValue = myUtils.captureThresTrackbarsValues()


    if opcServerEnabled:
        ladleLeftOPCVar.set_value(left_ladle_box.validated_number)
        ladleRightOPCVar.set_value(right_ladle_box.validated_number)

    left_ladle_img = left_ladle_box.detect(img)
    right_ladle_img = right_ladle_box.detect(img)
    img_result = left_ladle_box.draw(img)
    img_result = right_ladle_box.draw(img_result)
    # print(left_ladle_box.title, left_ladle_box.value, ', Validated is ', left_ladle_box.validated_number, ', List is ', left_ladle_box.value_validation_list)
    # print(right_ladle_box.title, right_ladle_box.value, ', Validated is ', right_ladle_box.validated_number, ', List is ', right_ladle_box.value_validation_list)

    cv2.imshow('Left Ladle Box', left_ladle_img)
    cv2.imshow('Right Ladle Box', right_ladle_img)
    cv2.imshow('digit1',left_ladle_box.left_digit_img)
    cv2.imshow('digit2', left_ladle_box.right_digit_img)
    cv2.imshow('digit3', right_ladle_box.left_digit_img)
    cv2.imshow('digit4', right_ladle_box.right_digit_img)

    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(img_result,f'FPS {int(fps)}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    cv2.imshow("Video Feed", img_result)
    cv2.setMouseCallback('Video Feed',mousePoints)


    if cv2.waitKey(parser.getint('video_feed','frame_delay',fallback=500)) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        # sys.exit()
        server.stop()
        quit()