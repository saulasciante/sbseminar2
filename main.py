import cv2.cv2 as cv

RECTANGLE_WIDTH = 2

left_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_rightear.xml')

cam = cv.VideoCapture(0)

while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    left_ears = left_ear_cascade.detectMultiScale(gray, 1.3, 5)
    right_ears = right_ear_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in left_ears:
        frame = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), RECTANGLE_WIDTH)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    for (x, y, w, h) in right_ears:
        img = cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), RECTANGLE_WIDTH)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv.imshow('img', img)
    cv.waitKey()

cam.release()
cv.destroyAllWindows()