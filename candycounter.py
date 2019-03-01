import sys

import cv2 as cv
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import colors
#from matplotlib import cm
#from sklearn.cluster import MiniBatchKMeans

kernel = np.ones((5,5), np.uint8)
image = None
colorVals = {
    'red': [130, 110, 220],
    'blue': [255, 165, 0],
    'green': [150, 220, 0],
    'orange': [30, 110, 250],
    'brown': [100, 90, 100],
    'yellow': [64, 240, 240]
}

colorTotals = {
    'red': 0, 
    'blue': 0,
    'green': 0, 
    'orange': 0,
    'brown': 0,

    'yellow': 0
}

def getColorValue(x, y, img):
    global colorVals
    current = None
    red = 255
    blue = 255
    green = 255

    pixel = img[y][x]
    for i in colorVals:
        compare = colorVals[i]
        reds = abs(compare[0] - pixel[0])
        blues = abs(compare[1] - pixel[1])
        greens = abs(compare[2] - pixel[2])
        if reds + blues + greens < red + blue + green:
            red = reds
            blue = blues
            green = greens
            current = i
    cv.putText(img, current, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return current

def countColors(img, pts):
    global colorTotals
    for i in pts:
        current = getColorValue(i[0], i[1], img)
        temp = colorTotals[current]
        colorTotals[current] = temp + 1
    for i in colorTotals:
        print(i + ": " + str(colorTotals[i]))

try:
    import easygui
except:
    print("Easygui not installed. To select images and videos via a graphic interface, please install easygui via 'pip install easygui'.")
    im = None
    if sys.version_info[0] < 3:
        im = raw_input("Enter filepath of an image: ")
    else:
        im = input("Enter filepath of an image: ")
    while True:
        try:
            image = cv.imread(im, cv.IMREAD_COLOR)
        except:
            print("Not a valid image.")
        else:
            break
else:
    import easygui
    image = easygui.fileopenbox(filetypes = ['*.jpg', '*.png', '*.mp4'])
    image = cv.imread(image, cv.IMREAD_COLOR)
    #print(image)
    #cv.imshow("File", image)
    
finally:
    cv.imshow("Original", image)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image = cv.GaussianBlur(image,(5,5),0)
    
    z = image.reshape((-1,3))
    z = np.float32(z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 16
    ret,label,center = cv.kmeans(z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    canny = cv.Canny(image, 150, 300)

    img1, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(res2, contours, -1, (0,255,0), -3)

    for i in contours:
        length = cv.arcLength(i, False)
        area = cv.contourArea(i)
        if area > 10:
            #print("Oh?")
            cv.fillPoly(canny, i, (255,255,255))
        if length > 15 and length < 300:
            '''x,y,w,h = cv.boundingRect(i)
            cv.rectangle(res2, (x,y),(x+w,y+h),(0,0,255),4)'''
            (x,y),radius = cv.minEnclosingCircle(i)
            center = (int(x),int(y))
            radius = int(0.8*radius)
            if radius > 3 and radius < 40:
                cv.circle(canny,center,radius,(255,255,255),-1)
            hull = cv.convexHull(i)
            M = cv.moments(hull)
            iX = int(M["m10"] / M["m00"])
            iY = int(M["m01"] / M["m00"])
            cv.circle(canny, (iX, iY), int(length*0.075), (255,255,255), -1)
    canny = cv.erode(canny, kernel, iterations=1)
    canny = cv.dilate(canny, kernel, iterations=1)
    canny = cv.erode(canny, kernel, iterations=1)
    
    img1, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    pts = list(())
    for i in contours:
        if cv.contourArea(i) > 5:
            M = cv.moments(i)
            iX = int(M["m10"] / M["m00"])
            iY = int(M["m01"] / M["m00"])
            pts.append([iX, iY])
            #cv.circle(hsv, (iX, iY), 7, (255, 0, 255), -1)
    
    countColors(image, pts)



    cv.imshow('Canny', canny)
    cv.imshow('HSV', hsv)
    cv.imshow("Image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()