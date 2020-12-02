import cv2.cv2 as cv
import numpy as np
import os

RECTANGLE_WIDTH = 2

left_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_rightear.xml')

# imageWithMask = cv.cvtColor(cv.imread('AWEForSegmentation/testannot_rect/0001.png'), cv.COLOR_BGR2GRAY)

def accuracy():
    # idea merge two pictures together and look colors
    pass

def maskCoordinates(image):
    mask = np.array(np.where(np.array(image) == 255))
    firstWhitePixel = mask[:, 0]
    lastWhitePixel = mask[:, -1]
    return firstWhitePixel[0], firstWhitePixel[1], lastWhitePixel[0], lastWhitePixel[1]


imageDirectory = 'AWEForSegmentation/test'

for imageFilename in os.listdir(imageDirectory):
    testImage = cv.imread(imageDirectory + '/' + imageFilename)
    testImageGray = cv.cvtColor(testImage, cv.COLOR_BGR2GRAY)

    left_ears = left_ear_cascade.detectMultiScale(testImageGray, 1.05, 4)
    right_ears = right_ear_cascade.detectMultiScale(testImageGray, 1.05, 4)

    for (x, y, w, h) in left_ears:
        img = cv.rectangle(testImage, (x, y), (x+w, y+h), (255, 0, 0), RECTANGLE_WIDTH)
        roi_gray = testImageGray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    for (x, y, w, h) in right_ears:
        img = cv.rectangle(testImage, (x, y), (x+w, y+h), (0, 255, 0), RECTANGLE_WIDTH)
        roi_gray = testImageGray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv.imshow('img', testImage)
    cv.waitKey(0)

cv.destroyAllWindows()
