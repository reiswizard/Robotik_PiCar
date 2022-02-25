import sys
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageGrab


def importModel():
    model = tf.keras.models.load_model('savedModels/trafficsignclassification2')
    return model


def classification(img):
    plt.figure(figsize=(10, 10))
    model = importModel()
    img = tf.image.resize(img, (32, 32))
    imageArray = tf.keras.utils.img_to_array(img)
    imageArray = tf.expand_dims(imageArray, 0)  # Create a batch
    predictions = model.predict(imageArray)
    score = tf.nn.softmax(predictions[0])
    # print(
    #     "The image belongs to {} with {:.2f} percent accurancy."
    #         .format(np.argmax(score), 100 * np.max(score))
    # )
    return np.argmax(score)


def detect_sign():
    while True:
        img = ImageGrab.grab(bbox=(200, 200, 840, 680))  # x, y, w, h
        img_np = np.array(img)
        src = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)
        # filename = 'test5.JPG'
        # src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)

        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)

        rows = gray.shape[0]
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                  param1=100, param2=30,
                                  minRadius=50, maxRadius=100)
        print(circles)
        circles_img = []
        if circles is not None:
            for circle in circles:
                for x, y, r in circle:
                    # print(int(x), int(y), int(r))
                    crop_img = src[int(y - r * 1.5):int(y + r * 1.5), int(x - r * 1.5):int(x + r * 1.5)]
                    circles_img.append(crop_img)
                    # hier kann das herausgeschnittene Bild an tenserflow methode übergeben werden.
                    predictedClass = classification(crop_img)
                    if predictedClass == 1 or predictedClass == 2:
                        print(predictedClass)

        # cv.imshow("detected circles", src)
        # cv.waitKey(0)
        cv.imshow("frame", src)
        cv.imshow("gray", gray)
        if cv.waitKey(1) & 0Xff == ord('q'):
            break

    # return predictedClass


if __name__ == "__main__":
    detect_sign()
