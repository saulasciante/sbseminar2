import cv2.cv2 as cv
import numpy as np
import os
from statistics import mean

RECTANGLE_WIDTH = 2


def detectionAccuracy(maskDetected, maskCorrect):
    blended = cv.addWeighted(maskDetected, 0.5, maskCorrect, 0.5, 0.0)
    blendedGrey = cv.cvtColor(blended, cv.COLOR_BGR2GRAY)

    intersection = np.sum(blendedGrey == 203)
    union = np.sum(blendedGrey == 128) + np.sum(blendedGrey == 75) + intersection
    accuracy = round(intersection/union, 3)

    # if accuracy > 0.8:
    #     cv.imshow('blended', blendedGrey)
    #     cv.waitKey(0)

    return accuracy


def maskCoordinates(image):
    mask = np.array(np.where(np.array(image) == 255))
    firstWhitePixel = mask[:, 0]
    lastWhitePixel = mask[:, -1]
    return firstWhitePixel[0], firstWhitePixel[1], lastWhitePixel[0], lastWhitePixel[1]


def detectEars(testImage):
    height, width, channels = testImage.shape
    testImageGray = cv.cvtColor(testImage, cv.COLOR_BGR2GRAY)

    left_ears = left_ear_cascade.detectMultiScale(testImageGray, 1.05, 3)
    right_ears = right_ear_cascade.detectMultiScale(testImageGray, 1.05, 3)

    maskDetected = np.zeros(shape=[height, width, channels], dtype=np.uint8)

    for (x, y, w, h) in left_ears:
        maskDetected = cv.rectangle(maskDetected, (x, y), (x + w, y + h), (0, 255, 0), -1)

    for (x, y, w, h) in right_ears:
        maskDetected = cv.rectangle(maskDetected, (x, y), (x + w, y + h), (0, 255, 0), -1)

    return maskDetected


if __name__ == '__main__':
    left_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_leftear.xml')
    right_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_rightear.xml')
    testImageDir = 'AWEForSegmentation/train/'
    maskImageDir = 'AWEForSegmentation/trainannot_rect/'
    testImageFilenames = sorted(os.listdir(testImageDir))
    accuracies = []

    for filename in testImageFilenames:
        testImage = cv.imread(testImageDir + filename)

        try:
            maskDetected = detectEars(testImage)
            maskCorrect = cv.imread(maskImageDir + filename)
            accuracy = detectionAccuracy(maskDetected, maskCorrect)
            accuracies.append(accuracy)
            print(filename, accuracy)
        except:
            print(f'There was an exception during detecting ears in file {filename}')

    print(f'Average detection accuracy: {mean(accuracies)}')
