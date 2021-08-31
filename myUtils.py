import cv2
import numpy as np
from configparser import ConfigParser



def empty(a):
    pass


def drawBoundingBox(img, bbox, colorBGR=(0, 255, 0), thickness=2, bias=0):
    cv2.rectangle(img, (bbox[0] - bias, bbox[1] - bias), (bbox[0] + bbox[2] + bias, bbox[1] + bbox[3] + bias), colorBGR,
                  2)


# 2 - CROPS AN IMAGE
def cropImage(img, x, y, height, width):
    imgCrop = img[y:y + height, x:x + width]
    return imgCrop


def createHSVTrackbars(windowName='HSVTrackBars', HueMin=0, HueMax=179, SatMin=0, SatMax=255, ValMin=0, ValMax=255):
    # CREATE NEW NAMED WINDOW
    cv2.namedWindow(windowName)
    cv2.resizeWindow(windowName, 640, 240)

    # CREATE HSV TRACKBARS

    cv2.createTrackbar("Hue Min", windowName, HueMin, 179, updateSaveTrackbarsValues)
    cv2.createTrackbar("Hue Max", windowName, HueMax, 179, updateSaveTrackbarsValues)
    cv2.createTrackbar("Sat Min", windowName, SatMin, 255, updateSaveTrackbarsValues)
    cv2.createTrackbar("Sat Max", windowName, SatMax, 255, updateSaveTrackbarsValues)
    cv2.createTrackbar("Val Min", windowName, ValMin, 255, updateSaveTrackbarsValues)
    cv2.createTrackbar("Val Max", windowName, ValMax, 255, updateSaveTrackbarsValues)


def captureHSVTrackbarValues(windowName='HSVTrackBars'):
    # CAPTURE TRACKBAR POSITION VALUES

    h_min = cv2.getTrackbarPos("Hue Min", windowName)
    h_max = cv2.getTrackbarPos("Hue Max", windowName)
    s_min = cv2.getTrackbarPos("Sat Min", windowName)
    s_max = cv2.getTrackbarPos("Sat Max", windowName)
    v_min = cv2.getTrackbarPos("Val Min", windowName)
    v_max = cv2.getTrackbarPos("Val Max", windowName)
    lowerb = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upperb = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lowerb, upperb  # ,h_min,h_max,s_min,s_max,v_min,v_max


def createThresTrackbars(windowName='ThresholdTrackBars', ThresLeft=127, ThresRight=127):
    # CREATE NEW NAMED WINDOW
    cv2.namedWindow(windowName)
    cv2.resizeWindow(windowName, 640, 240)

    # CREATE THRESHOLD TRACKBARS

    cv2.createTrackbar("Thres Left", windowName, ThresLeft, 255, updateSaveTrackbarsValues)
    cv2.createTrackbar("Thres Right", windowName, ThresRight, 255, updateSaveTrackbarsValues)


def captureThresTrackbarsValues(windowName='ThresholdTrackBars'):
    # CAPTURE TRACKBAR POSITION VALUES

    thr_left = cv2.getTrackbarPos("Thres Left", windowName)
    thr_right = cv2.getTrackbarPos("Thres Right", windowName)

    return thr_left, thr_right


def updateSaveTrackbarsValues(windowName='ThresholdTrackBars'):
    config = ConfigParser()
    config.read('config.ini')
    # SAVE HSV FILTER TRACK BAR VALUE
    config.set('HSV_Filter','huemin',str(cv2.getTrackbarPos('Hue Min','HSVTrackBars')))
    config.set('HSV_Filter', 'huemax',str(cv2.getTrackbarPos('Hue Max', 'HSVTrackBars')))
    config.set('HSV_Filter','satmin',str(cv2.getTrackbarPos('Sat Min','HSVTrackBars')))
    config.set('HSV_Filter', 'satmax',str(cv2.getTrackbarPos('Sat Max', 'HSVTrackBars')))
    config.set('HSV_Filter','valmin',str(cv2.getTrackbarPos('Val Min','HSVTrackBars')))
    config.set('HSV_Filter', 'valmax',str(cv2.getTrackbarPos('Val Max', 'HSVTrackBars')))

    # SAVE THRESHOLD TRACK BAR VALUE
    config.set('image_processing', 'thresholdladleleft',str(cv2.getTrackbarPos('Thres Left','ThresholdTrackBars')))
    config.set('image_processing', 'thresholdladleright',str(cv2.getTrackbarPos('Thres Right','ThresholdTrackBars')))

    with open('config.ini', 'w') as f:
        config.write(f)
    print("Saved Values!")


def removeBadContours(img, contours):
    mask = np.zeros(img.shape, dtype="uint8")
    print(img)
    cv2.imshow("removedbefore", img)
    for cnt in contours:
        bbox = cv2.boundingRect(cnt)
        bboxArea = bbox[2] * bbox[3]
        if bboxArea > 100000 and bboxArea < 300000:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            imgMasked = cv2.bitwise_and(img, img, mask=mask)
            cv2.imshow("removed", imgMasked)
    return img


def findNumberMajority(numberList):
    i = 0
    m = 0
    for cnt in range(0, 1):
        for x in numberList:
            if i == 0:
                m = x
                i = 1
            elif m == x:
                i += i
            else:
                i = i - 1
    return m


def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf
