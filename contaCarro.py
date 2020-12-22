import cv2 as cv
import numpy as np

class Ponto:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Sensor:
    def __init__(self, Ponto1, Ponto2, frameWeight, videoTam):
        self.Ponto1 = Ponto1
        self.Ponto2 = Ponto2
        self.frameWeight = frameWeight
        self.videoTam = videoTam
        self.mask = np.zeros((frameWeight, videoTam, 1), np.uint8)*abs(self.Ponto2.y - self.Ponto1.y)
        self.fullMaskArea = abs(self.Ponto2.x - self.Ponto1.x)
        cv.rectangle(self.mask, (self.Ponto1.x, self.Ponto1.y), (self.Ponto2.x, self.Ponto2.y), (255), thickness = cv.FILLED)
        self.stuation = False
        self.numCar = 0

video = cv.VideoCapture("video.mp4")
ret, frame = video.read()
croppedImage = frame[0:450, 0:450]
fgbg = cv.createBackgroundSubtractorMOG2()
Sensor1 = Sensor(Ponto(110, croppedImage.shape[1] - 35),
                 Ponto(250, croppedImage.shape[1] - 30),
                 croppedImage.shape[0],
                 croppedImage.shape[1])

kernel = np.ones((5, 5), np.uint8)
font = cv.FONT_HERSHEY_TRIPLEX
while ( 1 ):
    ret, frame = video.read()
    croppedImage = frame[0:450, 510:960]
    deletedBg = fgbg.apply(croppedImage)
    opImage = cv.morphologyEx(deletedBg, cv.MORPH_OPEN, kernel)
    ret, opImage = cv.threshold(opImage, 125, 255, cv.THRESH_BINARY)

    cnts, _ = cv.findContours(opImage, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    result = croppedImage.copy()

    zeros_image = np.zeros((croppedImage.shape[0], croppedImage.shape[1], 1), np.uint8)

    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        if ( w > 50 and h > 30  and w < 120 and h < 100 ):
            cv.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), thickness = 2)
            cv.rectangle(zeros_image, (x, y), (x + w, y + h), (255), thickness = cv.FILLED)

    mask1 = np.zeros((zeros_image.shape[0], zeros_image.shape[1], 1), np.uint8)
    maskResult = cv.bitwise_or(zeros_image, zeros_image, mask = Sensor1.mask)
    numCel = np.sum(maskResult == 255)

    sensorRate = numCel/Sensor1.fullMaskArea

    if sensorRate > 0:
        print("result : ", sensorRate, " situação: ", Sensor1.stuation)

    if ( sensorRate >= 0.9 and  sensorRate < 3.1 and Sensor1.stuation == False ):
        cv.rectangle(result, (Sensor1.Ponto1.x, Sensor1.Ponto1.y), (Sensor1.Ponto2.x, Sensor1.Ponto2.y),
                      (0, 255, 0,), thickness = cv.FILLED)
        Sensor1.stuation = True
    
    elif ( sensorRate < 0.9 and Sensor1.stuation == True ) :
        cv.rectangle(result, (Sensor1.Ponto1.x, Sensor1.Ponto1.y), (Sensor1.Ponto2.x, Sensor1.Ponto2.y),
                      (0, 0, 255), thickness = cv.FILLED)
        Sensor1.stuation = False
        Sensor1.numCar += 1
    
    else :
        cv.rectangle(result, (Sensor1.Ponto1.x, Sensor1.Ponto1.y), (Sensor1.Ponto2.x, Sensor1.Ponto2.y),
                      (0, 0, 255), thickness = cv.FILLED)

    cv.putText(result, str(Sensor1.numCar), (0, 100), font, 2, (0, 0, 0))

    cv.imshow("video", result)
    #cv.imshow("maskResult", maskResult)
    #cv.imshow("zeros_image", zeros_image)
    #cv.imshow("opImage", opImage)

    k = cv.waitKey(30) & 0xff
    if k == 27 :
        break

video.release()
cv.destroyAllWindows()