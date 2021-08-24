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
    # print(objects)
    # ladlesOPCObj = node.add_object('ns=2; s="Ladle Number"','Ladle Numbers')
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

### CONVOLUTIONAL NETWORK SUPPORT FUNCTIONS ###
def preProcessCNNImg(img):
    img = np.asarray(img)
    img = cv2.resize(img, (32, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 32, 32, 1)
    img = img / 255
    return img

def imgCNNpredict(img):
    classIndex = 0
    img = preProcessCNNImg(img)
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = int(np.amax(predictions) * 100)

    return classIndex, probVal

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


# thresholdladleLeft = 220
# thresholdladleRight = 180
thresholdladleLeft = parser.getint('image_processing','thresholdladleleft')
thresholdladleRight = parser.getint('image_processing','thresholdladleright')


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
# colorHSVFilter = (0,179,0,91,33,255)  # PINK PRESET HSV RANGE
# colorHSVFilter = (0,179,0,72,65,176)  # NUMBER COLOR PRESET HSV RANGE

colorHSVFilter = (parser.getint('HSV_Filter','huemin'),
                  parser.getint('HSV_Filter','huemax'),
                  parser.getint('HSV_Filter','satmin'),
                  parser.getint('HSV_Filter','satmax'),
                  parser.getint('HSV_Filter','valmin'),
                  parser.getint('HSV_Filter','valmax'))  # NUMBER COLOR PRESET HSV RANGE


# maxNumberSamples = 50  # SAMPLES TO TAKE BEFORE DECIDING THE LADLE NUMBER IS CORRECT, ASK MANY ANSWER ONCE
maxNumberSamples = parser.getint('neural_network','validationsamples')  # SAMPLES TO TAKE BEFORE DECIDING THE LADLE NUMBER IS CORRECT, ASK MANY ANSWER ONCE

ladleLeftNumberSamples = []
ladleRightNumberSamples = []


### CREATE AND INITIALISE HSV TRACKBARS

myUtils.createHSVTrackbars("HSVTrackBars",HueMin=colorHSVFilter[0],HueMax=colorHSVFilter[1],SatMin=colorHSVFilter[2],
                           SatMax=colorHSVFilter[3],ValMin=colorHSVFilter[4],ValMax=colorHSVFilter[5])

### CREATE AND INITIALISE THRESOLDING ADJUST TRACKBARS

myUtils.createThresTrackbars("ThresholdTrackBars",thresholdladleLeft,thresholdladleRight)


# url = "rtsp://10.81.98.80/?line=4?inst=2"
# url = "rtsp://10.81.98.165/?line=1"
url = parser.get('video_feed','url')

cap = cv2.VideoCapture(url)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open camera feed")

ret, img = cap.read()
print(img.shape)


def preProcessImg(img, x, y, h, w, scaleFactor=1, thresholdSet=128):
    kernel = np.ones((3, 3), np.uint8)
    pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    pts2 = np.float32(
        [[0, 0], [0 + w * scaleFactor, 0], [0, 0 + h * scaleFactor], [0 + w * scaleFactor, 0 + h * scaleFactor]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, matrix, (w * scaleFactor, h * scaleFactor))
    img = cv2.resize(img, (1000, 1000))
    img = myUtils.apply_brightness_contrast(img, -20, 30)
    # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21) ### THIS OPERATION IS CPU INTENSIVE
    # cv2.imshow("Contrast",img)
    # cv2.waitKey(2000)

    if colorFilterON:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # lHSV, uHSV = myUtils.captureHSVTrackbarValues()
        imgMask = cv2.inRange(img, lHSV, uHSV)
        imgMasked = cv2.bitwise_and(img, img, mask=imgMask)
        cv2.imshow("Masked Image",imgMasked)

        img = cv2.cvtColor(imgMasked,cv2.COLOR_HSV2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image",img)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    ret, img = cv2.threshold(img, thresholdSet, 255, cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,901,1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.bitwise_not(img)

    return img


def findNumberContours(image):
    detectedBboxNumbers = []
    detectedNumber = None

    imgCanny = cv2.Canny(image, 100, 150)
    imgContours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # imageNew = removeBadContours(image,contours=imgContours)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    imageCanvas = image.copy()
    # image = removeBadContours(image,imgContours)

    for cnt in imgContours:

        cv2.drawContours(imageCanvas, cnt, -1, (255, 0, 255), thickness=2)
        bbox = cv2.boundingRect(cnt)
        cv2.rectangle(imageCanvas, (bbox[0] - 5, bbox[1] - 5), (bbox[0] + bbox[2] + 5, bbox[1] + bbox[3] + 5),
                      (255, 0, 0), 2)
        bboxArea = bbox[3] * bbox[2]
        contourArea = cv2.contourArea(cnt)
        # print("Height= ", bbox[2])
        # print(bboxArea)

        fillDegree = contourArea / bboxArea
        # print(fillDegree)

        bboxAspectRatio = bbox[3] / bbox[2]
        ################################################ DOUBLE DIGIT DETECTION
        if minDblDigitBoxWidth < bbox[2] < maxDblDigitBoxWidth and minDblDigitBoxHeigth < bbox[
            3] < maxDblDigitBoxHeigth and minFillDregree < fillDegree < maxFillDegree:
            myUtils.drawBoundingBox(imageCanvas, bbox, colorBGR=(0, 255, 255), thickness=2, bias=5)
            ## IF SO, SPLIT BOX IN HALF AND FEED TO CNN
            x, y, w, h = bbox
            bboxMiddlePoint = w // 2
            bbox1 = x, y, bboxMiddlePoint, h  ### LEFT DIGIT
            detectedBboxNumbers.append(bbox1)
            myUtils.drawBoundingBox(imageCanvas, bbox1, (255, 0, 255), 2, -2)
            bbox2 = x + bboxMiddlePoint, y, w // 2, h  ### RIGHT DIGIT
            detectedBboxNumbers.append(bbox2)
            myUtils.drawBoundingBox(imageCanvas, bbox2, (255, 0, 255), 2, -2)

        ############################################### SINGLE DIGIT DETECTION

        # if bboxAspectRatio > 1.2 and bboxArea > 30000 and bboxArea < 100000 and bbox[3] < 300 and fillDegree <0.9:
        if minSnglDigitBoxHeigth < bbox[3] < maxSnglDigitBoxHeigth and minSnglDigitBoxWidth < bbox[
            2] < maxSnglDigitBoxWidth:  # and minFillDregree < fillDegree < maxFillDegree:
            # print(fillDegree)
            detectedBboxNumbers.append(bbox)

            # print("Height= ", bbox[3])
            # print("Bounding Box area = ",bboxArea)
            # print("Contour Area = ",contourArea)
            cv2.rectangle(imageCanvas, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
            ### EXTRACT IMAGE ON BOUNDING BOX
            # imgCNNpredict(img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])
            # cv2.imshow("Clipped",image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]])

            # print(imgCNNpredict(image[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]))
            # print(len(detectedBboxNumbers))
            # cv2.waitKey(2000)
        if len(detectedBboxNumbers) == 2:
            # Two bounding boxes containing a number are available
            # print(detectedBboxNumbers)
            # print(detectedBboxNumbers.sort())
            if detectedBboxNumbers[0][0] < detectedBboxNumbers[1][0]:
                A, probA = imgCNNpredict(
                    image[detectedBboxNumbers[0][1]:detectedBboxNumbers[0][1] + detectedBboxNumbers[0][3],
                    detectedBboxNumbers[0][0]:detectedBboxNumbers[0][0] + detectedBboxNumbers[0][2]])
                B, probB = imgCNNpredict(
                    image[detectedBboxNumbers[1][1]:detectedBboxNumbers[1][1] + detectedBboxNumbers[1][3],
                    detectedBboxNumbers[1][0]:detectedBboxNumbers[1][0] + detectedBboxNumbers[1][2]])
            else:
                B, probB = imgCNNpredict(
                    image[detectedBboxNumbers[0][1]:detectedBboxNumbers[0][1] + detectedBboxNumbers[0][3],
                    detectedBboxNumbers[0][0]:detectedBboxNumbers[0][0] + detectedBboxNumbers[0][2]])
                A, probA = imgCNNpredict(
                    image[detectedBboxNumbers[1][1]:detectedBboxNumbers[1][1] + detectedBboxNumbers[1][3],
                    detectedBboxNumbers[1][0]:detectedBboxNumbers[1][0] + detectedBboxNumbers[1][2]])
            if probA > cnnCertainty and probB > cnnCertainty:
                detectedNumber = (int(str(A) + str(B)))
            detectedBboxNumbers = []
            break

    return imageCanvas, detectedNumber


### MAIN LOOP ###
while True:
    ret, img = cap.read()
    # img = cv2.imread('Resources/PinkTest2.png')
    if ret == False:
        continue

    lHSV, uHSV = myUtils.captureHSVTrackbarValues()
    thresholdladleLeft,thresholdladleRight = myUtils.captureThresTrackbarsValues()

    imgLadleLeft = preProcessImg(img, x1, y1, h1, w1, scaleFactor, thresholdSet=thresholdladleLeft)
    imgLadleRight = preProcessImg(img, x2, y2, h2, w2, scaleFactor, thresholdSet=thresholdladleRight)

    imgLeftLadleDetection, ladleLeftNumber = findNumberContours(imgLadleLeft)
    imgRightLadleDetection, ladleRightNumber = findNumberContours(imgLadleRight)

    ladleLeftNumberSamples.append(ladleLeftNumber)
    validatedLeftLadleNumber = myUtils.findNumberMajority(ladleLeftNumberSamples)
    print(ladleLeftNumberSamples, validatedLeftLadleNumber)
    ladleRightNumberSamples.append(ladleRightNumber)
    validatedRightLadleNumber = myUtils.findNumberMajority(ladleRightNumberSamples)
    print(ladleRightNumberSamples, validatedRightLadleNumber)

    if len(ladleLeftNumberSamples) >= maxNumberSamples:
        ladleLeftNumberSamples.pop(0)
    if len(ladleRightNumberSamples) >= maxNumberSamples:
        ladleRightNumberSamples.pop(0)

    if ladleLeftNumber == None and leftScanEnabled:
        ### MOVE BOX IF NUMBER HAS NOT BEEN FOUND (SCAN FOR IT)
        for verMove in range(img.shape[0] // 2, img.shape[0] - (img.shape[0] // 3), h1 // 2):
            for horMove in range(0, (img.shape[1] // 2) - w1, w1 // 2):
                x1, y1 = horMove, verMove

                imgLadleLeft = preProcessImg(img, x1, y1, h1, w1, scaleFactor, thresholdSet=thresholdladleLeft)
                imgLeftLadleDetection, ladleLeftNumber = findNumberContours(imgLadleLeft)
                if ladleLeftNumber != None:
                    break
                imgResult = img.copy()
                cv2.rectangle(imgResult, (x1, y1), (x1 + w1, y1 + w1), (0, 255, 0), 2)
                cv2.putText(imgResult, "Left Ladle => " + str(ladleLeftNumber), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (0, 255, 0), 2)
                cv2.imshow("Ladle Left", imgLeftLadleDetection)
                cv2.imshow("Video Feed", imgResult)

                # cv2.waitKey(100)
            if ladleLeftNumber != None:
                break

    if ladleRightNumber == None and rightScanEnabled:
        ### MOVE BOX IF NUMBER HAS NOT BEEN FOUND (SCAN FOR IT)
        for verMove in range(img.shape[0] // 2, img.shape[0] - (img.shape[0] // 3), h2 // 2):
            for horMove in range(img.shape[1] // 2, (img.shape[1]) - w2, w2 // 2):
                x2, y2 = horMove, verMove

                imgLadleRight = preProcessImg(img, x2, y2, h2, w2, scaleFactor, thresholdSet=thresholdladleLeft)
                imgRightLadleDetection, ladleRightNumber = findNumberContours(imgLadleRight)
                if ladleRightNumber != None:
                    break
                imgResult = img.copy()
                cv2.rectangle(imgResult, (x2, y2), (x2 + w2, y2 + w2), (0, 0, 255), 2)
                cv2.putText(imgResult, "Right Ladle => " + str(validatedRightLadleNumber), (x2, y2 - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 0, 255), 2)
                cv2.imshow("Ladle Right", imgRightLadleDetection)
                cv2.imshow("Video Feed", imgResult)

                cv2.waitKey(100)
            if ladleRightNumber != None:
                break
    if opcServerEnabled:
        ladleLeftOPCVar.set_value(validatedLeftLadleNumber)
        ladleRightOPCVar.set_value(validatedRightLadleNumber)

    print("Left ladle is = ", ladleLeftNumber)
    print("Right ladle is = ", ladleRightNumber)

    ### NEURAL NETWORK
    # classIndexLeft, probValLeft = imgCNNpredict(imgLadleLeft)
    # classIndexRight, probValRight = imgCNNpredict(imgLadleRight)

    # print(classIndexLeft, probValLeft)
    # print(classIndexRight, probValRight)

    # imgRGB = cv2.cvtColor(imgLadleLeft, cv2.COLOR_BGR2RGB)


    cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + w1), (0, 255, 0), 2)
    cv2.putText(img, "Left Ladle => " + str(validatedLeftLadleNumber), (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 255, 0), 2)
    cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + w2), (0, 0, 255), 2)
    cv2.putText(img, "Right Ladle => " + str(validatedRightLadleNumber), (x2, y2 - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 0, 255), 2)
    cv2.imshow("Ladle Left", imgLeftLadleDetection)
    cv2.imshow("Ladle Right", imgRightLadleDetection)
    cv2.imshow("Video Feed", img)
    if cv2.waitKey(parser.getint('video_feed','frame_delay',fallback=500)) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        # sys.exit()
        server.stop()
        quit()