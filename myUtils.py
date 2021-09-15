import cv2
import numpy as np
from configparser import ConfigParser
import pickle


def empty(a):
    pass


def drawBoundingBox(img, bbox, colorBGR=(0, 255, 0), thickness=2, bias=0):
    cv2.rectangle(img, (bbox[0] - bias, bbox[1] - bias), (bbox[0] + bbox[2] + bias, bbox[1] + bbox[3] + bias), colorBGR,
                  2)


def draw_text_on_top_left_corner(image, text=()):
    padding = 10
    v_offset = 0
    line_spacing = 20
    curr_line = 0

    for line in text:
        curr_line += line_spacing
        cv2.putText(image, line, (padding, curr_line + v_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image


# 2 - CROPS AN IMAGE
def cropImage(img, x, y, height, width):
    imgCrop = img[y:y + height, x:x + width]
    return imgCrop


def createHSVTrackbars(windowName='HSVTrackBars', HueMin=0, HueMax=179, SatMin=0, SatMax=255, ValMin=0, ValMax=255):
    # CREATE NEW NAMED WINDOW
    cv2.namedWindow(windowName)
    cv2.resizeWindow(windowName, 640, 240)

    # CREATE HSV TRACKBARS

    cv2.createTrackbar("Hue Min", windowName, HueMin, 179, updateSaveHSVTrackbarsValues)
    cv2.createTrackbar("Hue Max", windowName, HueMax, 179, updateSaveHSVTrackbarsValues)
    cv2.createTrackbar("Sat Min", windowName, SatMin, 255, updateSaveHSVTrackbarsValues)
    cv2.createTrackbar("Sat Max", windowName, SatMax, 255, updateSaveHSVTrackbarsValues)
    cv2.createTrackbar("Val Min", windowName, ValMin, 255, updateSaveHSVTrackbarsValues)
    cv2.createTrackbar("Val Max", windowName, ValMax, 255, updateSaveHSVTrackbarsValues)


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
    return lowerb, upperb, [h_min, h_max, s_min, s_max, v_min, v_max]


def createThresTrackbars(windowName='ThresholdTrackBars', ThresLeft=127, ThresRight=127):
    # CREATE NEW NAMED WINDOW
    cv2.namedWindow(windowName)
    cv2.resizeWindow(windowName, 640, 240)

    # CREATE THRESHOLD TRACKBARS

    cv2.createTrackbar("Thres Left", windowName, ThresLeft, 255, updateSaveThreshTrackbarsValues)
    cv2.createTrackbar("Thres Right", windowName, ThresRight, 255, updateSaveThreshTrackbarsValues)


def captureThresTrackbarsValues(windowName='ThresholdTrackBars'):
    # CAPTURE TRACKBAR POSITION VALUES

    thr_left = cv2.getTrackbarPos("Thres Left", windowName)
    thr_right = cv2.getTrackbarPos("Thres Right", windowName)

    return thr_left, thr_right


def updateSaveHSVTrackbarsValues(windowName='ThresholdTrackBars'):
    config = ConfigParser()
    config.read('config.ini')
    # SAVE HSV FILTER TRACK BAR VALUE
    config.set('HSV_Filter', 'huemin1', str(cv2.getTrackbarPos('Hue Min', 'HSV Left Filter')))
    config.set('HSV_Filter', 'huemax1', str(cv2.getTrackbarPos('Hue Max', 'HSV Left Filter')))
    config.set('HSV_Filter', 'satmin1', str(cv2.getTrackbarPos('Sat Min', 'HSV Left Filter')))
    config.set('HSV_Filter', 'satmax1', str(cv2.getTrackbarPos('Sat Max', 'HSV Left Filter')))
    config.set('HSV_Filter', 'valmin1', str(cv2.getTrackbarPos('Val Min', 'HSV Left Filter')))
    config.set('HSV_Filter', 'valmax1', str(cv2.getTrackbarPos('Val Max', 'HSV Left Filter')))
    config.set('HSV_Filter', 'huemin2', str(cv2.getTrackbarPos('Hue Min', 'HSV Right Filter')))
    config.set('HSV_Filter', 'huemax2', str(cv2.getTrackbarPos('Hue Max', 'HSV Right Filter')))
    config.set('HSV_Filter', 'satmin2', str(cv2.getTrackbarPos('Sat Min', 'HSV Right Filter')))
    config.set('HSV_Filter', 'satmax2', str(cv2.getTrackbarPos('Sat Max', 'HSV Right Filter')))
    config.set('HSV_Filter', 'valmin2', str(cv2.getTrackbarPos('Val Min', 'HSV Right Filter')))
    config.set('HSV_Filter', 'valmax2', str(cv2.getTrackbarPos('Val Max', 'HSV Right Filter')))
    with open('config.ini', 'w') as f:
        config.write(f)


def updateSaveThreshTrackbarsValues(windowName='ThresholdTrackBars'):
    config = ConfigParser()
    config.read('config.ini')
    # SAVE THRESHOLD TRACK BAR VALUE
    config.set('image_processing', 'thresholdladleleft', str(cv2.getTrackbarPos('Thres Left', 'ThresholdTrackBars')))
    config.set('image_processing', 'thresholdladleright', str(cv2.getTrackbarPos('Thres Right', 'ThresholdTrackBars')))

    with open('config.ini', 'w') as f:
        config.write(f)


def updateSaveBoxPosition(box_list):
    config = ConfigParser()
    config.read('config.ini')
    config.set('box_coordinates', 'x1', str(box_list[0].pos_xy[0]))
    config.set('box_coordinates', 'y1', str(box_list[0].pos_xy[1]))
    config.set('box_coordinates', 'x2', str(box_list[1].pos_xy[0]))
    config.set('box_coordinates', 'y2', str(box_list[1].pos_xy[1]))
    with open('config.ini', 'w') as f:
        config.write(f)


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


class DetectionBox:

    def __init__(self, x=0, y=0, w=100, h=100, title="", color=(0, 255, 0)):
        self.pos_xy = [x, y]
        self.size = [w, h]
        self.image = None
        self.value = None
        self.left_digit = None
        self.right_digit = None
        self.left_digit_img = np.array((w, h, 3), np.uint8)
        self.right_digit_img = np.array((w, h, 3), np.uint8)
        self.thresholdValue = 128
        self.title = title
        self.thickness = 2
        self.title_thickness = 2
        self.validated_number = None
        self.validation_sample_target = 50
        self.selected = False
        self.show_controls = True
        self.selected_offset = [0, 0]
        self.value_validation_list = []
        self.left_digit_probability = 0
        self.right_digit_probability = 0
        self.cnn_certainty = 0.9
        self.pt1 = [self.pos_xy[0], self.pos_xy[1]]
        self.pt2 = [self.pos_xy[0] + self.size[0], self.pos_xy[1] + self.size[1]]
        self.box = [self.pt1, self.pt2, w, h]
        self.color = color
        self.corner_color = self.color
        self.colorHSVFilter = [0, 179, 0, 255, 0, 255]
        ### LOAD CONVOLUTIONAL NEURAL NETWORK MODEL
        pickle_in = open("model_trained_20.p", "rb")
        self.model = pickle.load(pickle_in)
        self.min_sngl_digit_box_width = 100
        self.max_sngl_digit_box_width = 250
        self.min_sngl_digit_box_height = 200
        self.max_sngl_digit_box_height = 350

        self.min_dbl_digit_box_width = 320
        self.max_dbl_digit_box_width = 500

        self.min_dbl_digit_box_height = 250
        self.max_dbl_digit_box_height = 450

        self.min_fill_dregree = 0.2
        self.max_fill_dregree = 0.9

        self.show_processed_images = False

    def draw(self, image):

        cv2.rectangle(image, self.pt1, self.pt2, self.color, self.thickness)
        cv2.putText(image, f'{self.title} => ' + str(self.validated_number), (self.pos_xy[0], self.pos_xy[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, self.color, self.title_thickness)

        if self.selected:
            self.__draw_border(image, self.corner_color, 8, 5, 10)
        else:
            self.__draw_border(image, self.corner_color, 4, 5, 10)

        return image

    def __draw_border(self, image, color, thickness, r, d):
        x1, y1 = self.pt1
        x2, y2 = self.pt2
        # Top left
        cv2.line(image, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(image, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(image, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(image, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(image, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(image, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(image, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(image, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(image, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(image, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(image, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(image, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def update(self, x, y):
        self.pos_xy = x, y
        self.pt1 = [x, y]
        self.pt2 = [x + self.size[0], y + self.size[1]]

    def __validate_number(self, number, sample_target=50):
        self.value_validation_list.append(number)

        if len(self.value_validation_list) > sample_target: self.value_validation_list.pop(0)
        i = 0
        m = 0
        for cnt in range(0, 1):
            for x in self.value_validation_list:
                if i == 0:
                    m = x
                    i = 1
                elif m == x:
                    i += i
                else:
                    i = i - 1
        return m

    @staticmethod
    def draw_bounding_box(image, bbox, color_bgr=(0, 255, 0), thickness=2, bias=0):
        cv2.rectangle(image, (bbox[0] - bias, bbox[1] - bias), (bbox[0] + bbox[2] + bias, bbox[1] + bbox[3] + bias),
                      color_bgr, 2)

    def __auto_threshold(self):
        min_value = 10
        threshold_step = 1

        if self.selected: return

        if self.thresholdValue > 254: self.thresholdValue = min_value
        self.thresholdValue += threshold_step

    def __apply_brightness_contrast(self, input_img, brightness=0, contrast=0):
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

    def __pre_process_img(self, image, scaleFactor=1):
        """
        PRE-PROCESSES THE IMAGE, CROPS THE BOX AREA AND RESIZES IT FOR BETTER VIEWING
        A RESIZED AND HSV MASKED IMAGE IS RETURNED

        :param image: Source Image from Video Feed (BGR)
        :param scaleFactor: Usually 1, multiplier for resulting image size
        :return: Returns the cropped and processed image ready for digit separation and detection
        """
        x, y = self.pt1
        w, h = self.size
        kernel = np.ones((3, 3), np.uint8)  # Prepare kernel
        pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])  # Source points FROM for warping
        pts2 = np.float32([[0, 0], [0 + w * scaleFactor, 0], [0, 0 + h * scaleFactor],
                           [0 + w * scaleFactor, 0 + h * scaleFactor]])  # Warping destination points TO
        matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Get rtansformation matrix FROM->TO
        img = cv2.warpPerspective(image, matrix, (w * scaleFactor, h * scaleFactor))  # Warp/Crop image
        img = cv2.resize(img, (1000, 1000))  # Resize to 1000px by 1000px
        img = self.__apply_brightness_contrast(img, -20, 30)  # Apply brightness and contrast correction
        # img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21) ### THIS OPERATION IS CPU INTENSIVE

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # BGR to HSV for HSV Filtering mask
        l_hsv, u_hsv = self.__hsv_filter()  # Get HSV boundaries from class attributes fed by track bars
        img_mask = cv2.inRange(img, l_hsv, u_hsv)  # Apply in range for HSV filtering
        img_masked = cv2.bitwise_and(img, img, mask=img_mask)  # Apply HSV mask
        # cv2.imshow("Masked Image Class", imgMasked)

        img = cv2.cvtColor(img_masked, cv2.COLOR_HSV2BGR)  # Return to BGR color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Go B/W color space, flattens out, goes from 3 channels to 1 chnl
        # cv2.imshow("Gray Image", img)
        img = cv2.equalizeHist(img)  # Historise to balance shadows/lights
        img = cv2.GaussianBlur(img, (5, 5), 1)  # Applies gaussian blur
        ret, img = cv2.threshold(img, self.thresholdValue, 255, cv2.THRESH_BINARY)  # Thresholds, image is 1 chnl
        img = cv2.erode(img, kernel, iterations=1)  # Apply erosion
        img = cv2.bitwise_not(img)  # Reverses color, black over white

        return img  # Returns image inside box, cleaned and in 1 chnl ready to find contours

        ### CONVOLUTIONAL NETWORK SUPPORT FUNCTIONS ###

    @staticmethod
    def __pre_process_cnn_img(image):
        img = np.asarray(image)
        img = cv2.resize(img, (32, 32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        img = img.reshape(1, 32, 32, 1)
        img = img / 255
        return img

    def __img_cnn_predict(self, image):
        image = self.__pre_process_cnn_img(image)
        class_index = int(self.model.predict_classes(image))
        predictions = self.model.predict(image)
        prob_val = int(np.amax(predictions) * 100)

        return class_index, prob_val

    def __crop_image(self, image, box):
        x, y, w, h = box

        img_cropped = image[y:y + h, x:x + w]
        return img_cropped

    def __hsv_filter(self):
        """
        Assembles lower HSV boundary and upper HSV boundary color range for masking
        :return: lowerb and upperb containing two ranges of HSV colors
        """
        h_min = self.colorHSVFilter[0]
        h_max = self.colorHSVFilter[1]
        s_min = self.colorHSVFilter[2]
        s_max = self.colorHSVFilter[3]
        v_min = self.colorHSVFilter[4]
        v_max = self.colorHSVFilter[5]

        lowerb = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upperb = np.array([h_max, s_max, v_max], dtype=np.uint8)
        return lowerb, upperb

    def __update_value(self, value):
        self.value = value
        # Validate number by finding majority
        self.validated_number = self.__validate_number(value, sample_target=self.validation_sample_target)

    def detect_and_draw(self, image):
        self.detect(image)
        image = self.draw(image)
        return image

    def detect(self, image):
        """

        :param image: Full unmodified frame from Video Feed
        :return: Returns annotated image with drawn box and information about detected number
        """
        detected_box_numbers = []  # Stores box coordinates where potential digits are, after shape filters
        self.__update_value(None)  # Adds a None value to the voting list, if a number is detected later, this last
        # NONE is removed and replaced with the number

        img = self.__pre_process_img(image)  # Pass full unmodified frame, return extracted box image ready to find
        # contours, image returns with only 1 chnl, black over white background

        img_canny = cv2.Canny(img, 100, 150)  # Apple Canny to prepare for edge detection
        img_contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # Find contours
        # imageNew = removeBadContours(image,contours=imgContours)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Go back to BGR color space to be able to draw in colors
        image_canvas = img.copy()  # Makes a copy of the pre-processed frame to draw contours on, img is BGR 3ch
        # black over white
        # image = removeBadContours(image,imgContours)

        if len(img_contours) == 0: self.__auto_threshold()  # Automatically adjust threshold if no contours were detected
        for i, cnt in enumerate(img_contours):  # Process all contours one by one

            if i == len(img_contours)-1:
                self.__auto_threshold() # Automatically adjust threshold if reached the last contour without detections

            cv2.drawContours(image_canvas, cnt, -1, (255, 0, 255), thickness=2)  # Draw current contour in image_canvas
            bbox = cv2.boundingRect(cnt)  # Get current contour bounding box data
            cv2.rectangle(image_canvas, (bbox[0] - 5, bbox[1] - 5), (bbox[0] + bbox[2] + 5, bbox[1] + bbox[3] + 5),
                          (255, 0, 0), 2)  # Draw bounding box
            bbox_area = bbox[3] * bbox[2]  # Calculate contour bounding box area
            contour_area = cv2.contourArea(cnt)  # Get contour area

            fill_degree = contour_area / bbox_area  # Calculate how much of the contour is filling its bounding box

            bbox_aspect_ratio = bbox[2] / bbox[3]  # Calculates aspect ratio w/h, desired is shape similar to digit box
            # vertical rectangle

            # From here on, the current contour bounding box can be inspected in two ways, double or single digit

            # ================================ DOUBLE DIGIT DETECTION ================================
            # Extract the contour portion of the image and examine for digits, split in two
            # Filter contour bounding boxes in first IF, we're looking for a squarish shape with a certain size in the
            # image in where two digits would fit together
            if self.min_dbl_digit_box_width < bbox[2] < self.max_dbl_digit_box_width and \
                    self.min_dbl_digit_box_height < bbox[3] < self.max_dbl_digit_box_height and \
                    self.min_fill_dregree < fill_degree < self.max_fill_dregree:
                self.draw_bounding_box(image_canvas, bbox, color_bgr=(0, 255, 255), thickness=2, bias=5)
                # If here, the bbox passed the filter, SPLIT BOX IN HALF AND STORE BBOX DATA IN LIST
                x, y, w, h = bbox  # Separate the box data
                bbox_middle_point = w // 2  # Calculate horizontal half
                bbox1 = x, y, bbox_middle_point, h  # GET LEFT DIGIT, box's half left side
                detected_box_numbers.append(bbox1)  # Store left digit box data in list, one digit stored

                self.draw_bounding_box(image_canvas, bbox1, (255, 0, 255), 2, -2)  # Bbox passed filter and contains
                # possible left digit, draw a box around it on canvas to identify it was captured

                bbox2 = x + bbox_middle_point, y, w // 2, h  # GET RIGHT DIGIT, box's half right side
                detected_box_numbers.append(bbox2)  # Store right digit box data in list, two digits stored
                self.draw_bounding_box(image_canvas, bbox2, (255, 0, 255), 2, -2)  # Bbox passed filter and contains
                # possible right digit, draw a box around it on canvas to identify it was captured

            # ================================ SINGLE DIGIT DETECTION ================================
            # Extract the contour portion of the image and examine for digits
            # Filter contour bounding boxes in first IF, we're looking for a vertical rectangular shape with a certain
            # size in the image in where one digit would fit

            # if bboxAspectRatio > 1.2 and bboxArea > 30000 and bboxArea < 100000 and bbox[3] < 300 and fillDegree <0.9:

            if self.min_sngl_digit_box_height < bbox[3] < self.max_sngl_digit_box_height and \
                    self.min_sngl_digit_box_width < bbox[
                2] < self.max_sngl_digit_box_width:  # and minFillDregree < fillDegree < maxFillDegree:
                # If here, the bbox passed the filter, STORE SINGLE BBOX DATA IN LIST
                detected_box_numbers.append(bbox)  # Store digit box data in list, one digit stored at a time
                self.draw_bounding_box(image_canvas, bbox, (0, 255, 0), 2, -2)  # Bbox passed filter and contains
                # possible digit, draw a box around it on canvas to identify it was captured



            if len(detected_box_numbers) == 2:  # Triggers when two bboxes are in the list whose passed all filters
                # Two bounding boxes containing a potential digit are available
                detected_box_numbers.sort()  # Sort detected digit boxes so the lowest x is the left digit
                self.left_digit_img = self.__crop_image(img, detected_box_numbers[
                    0])  # Extract image portion containing left digit
                self.right_digit_img = self.__crop_image(img, detected_box_numbers[
                    1])  # Extract image portion containing right digit

                # Feed digit image to CNN and take detected value and probability
                self.left_digit, self.left_digit_probability = self.__img_cnn_predict(self.left_digit_img)
                # Feed digit image to CNN and take detected value and probability
                self.right_digit, self.right_digit_probability = self.__img_cnn_predict(self.right_digit_img)

                # Check if the certainties for both digits are higher than expected

                if self.left_digit_probability > self.cnn_certainty and self.right_digit_probability > self.cnn_certainty:
                    # self.value = (int(str(self.left_digit) + str(self.right_digit)))
                    self.value_validation_list.pop(
                        -1)  # Digits were found, remove last None and replace with current value
                    self.__update_value(self.left_digit * 10 + self.right_digit)  # Store number if it was recognised ok

                detected_box_numbers = []  # Clear detected boxes for new detection if two boxes where in
                break  # Exit and stop looking for contours, we already found two, wait for next frame
        self.image = image_canvas.copy()  # Copy the annotated image on the class
        if self.show_processed_images:
            cv2.imshow(self.title, image_canvas)
            cv2.imshow(f'{self.title} - Left digit', self.left_digit_img)
            cv2.imshow(f'{self.title} - Right digit', self.right_digit_img)
        else:
            try:
                cv2.destroyWindow(self.title)
            except:
                pass

            try:
                cv2.destroyWindow(f'{self.title} - Left digit')
            except:
                pass
            try:
                cv2.destroyWindow(f'{self.title} - Right digit')
            except:
                pass

        return image_canvas
