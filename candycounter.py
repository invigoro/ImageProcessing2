#@authors Cory Peterson and Timothy Wells
#VIDEO WORKS

import sys

import cv2 as cv
import numpy as np


kernel = np.ones((5,5), np.uint8) 
image = None    #actual image file/frame of image
im = None   #image filename

#keep dict for comparing color values
colorVals = {
    'red': [130, 110, 220],
    'blue': [255, 165, 0],
    'green': [150, 220, 0],
    'orange': [30, 110, 250],
    'brown': [100, 90, 100],
    'yellow': [64, 240, 240]
}

#count of each color of candy
colorTotals = {
    'red': 0, 
    'blue': 0,
    'green': 0, 
    'orange': 0,
    'brown': 0,
    'yellow': 0
}

#find closest approximate color from dict
def getColorValue(x, y, img, labeled):
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
    cv.putText(labeled, current, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return current

#count m&ms by color and label them
def countColors(img, pts, orig, labeled):
    global colorTotals
    for i in pts:
        current = getColorValue(i[0], i[1], img, labeled)
        temp = colorTotals[current]
        colorTotals[current] = temp + 1
    height = len(orig)
    iterator = 0
    cv.rectangle(orig, (0,height), (130, height - 160), (255, 255, 255), -1)
    for i in colorTotals:
        color = colorVals[i]
        iterator += 22
        print(i + ": " + str(colorTotals[i]))
        cv.putText(orig, str(i + ": " + str(colorTotals[i])), (10, height - iterator), cv.FONT_HERSHEY_SIMPLEX, 0.65, (color[0], color[1], color[2]), 2)

def resetVals():
    global colorTotals
    for i in colorTotals:
        colorTotals[i] = 0

#main function for image manipulation
def findCandy(image):
    resetVals()
    originalImage = image.copy()
    labeled = image.copy()
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image = cv.GaussianBlur(image,(5,5),0)
    
    #reduce colors in image to make color detection easier
    z = image.reshape((-1,3))
    z = np.float32(z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 16
    ret,label,center = cv.kmeans(z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    #detect edges
    canny = cv.Canny(image, 150, 300)

    #contours for finding circles
    img1, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(res2, contours, -1, (0,255,0), -3)

    for i in contours:

        #length for non-closed contours, area for closed
        length = cv.arcLength(i, False)
        area = cv.contourArea(i)

        #fill shape if it ain't tiny
        if area > 10:
            cv.fillPoly(canny, i, (255,255,255))

        #only get open shapes in a certain range to filter out grain and the background
        if length > 15 and length < 300:
            (x,y),radius = cv.minEnclosingCircle(i)
            center = (int(x),int(y))
            radius = int(0.8*radius)
            #draw circles at the min enclosing circle and center of mass of convex hull in order to best approximate center of circle
            #overlapping circles will either be joined together of split in the next step
            if radius > 3 and radius < 40:
                cv.circle(canny,center,radius,(255,255,255),-1)
            hull = cv.convexHull(i)
            M = cv.moments(hull)
            iX = int(M["m10"] / M["m00"])
            iY = int(M["m01"] / M["m00"])
            cv.circle(canny, (iX, iY), int(length*0.075), (255,255,255), -1)

    #erase tiny particles, join close blobs, and split again
    canny = cv.erode(canny, kernel, iterations=1)
    canny = cv.dilate(canny, kernel, iterations=1)
    canny = cv.erode(canny, kernel, iterations=1)
    
    #these contours represent the actual centers of candies
    img1, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    #start drawing the candy centers and add them to a list of points to count
    pts = list(())
    for i in contours:
        if cv.contourArea(i) > 5:
            M = cv.moments(i)
            iX = int(M["m10"] / M["m00"])
            iY = int(M["m01"] / M["m00"])
            pts.append([iX, iY])
            cv.circle(hsv, (iX, iY), 7, (255, 0, 255), -1)

    #find colors of each point and label them
    countColors(image, pts, originalImage, labeled)

    cv.imshow('Canny', canny)
    cv.imshow('HSV', hsv)
    cv.imshow("Blurred", image)
    cv.imshow("Colors Found", originalImage)
    cv.imshow("Labeled", labeled)
    cv.imshow('Res2', res2)


try:
    import easygui
except:
    print("Easygui not installed. To select images and videos via a graphic interface, please install easygui via 'pip install easygui'.")
    if sys.version_info[0] < 3:
        im = raw_input("Enter filepath of an image: ")
    else:
        im = input("Enter filepath of an image: ")
    while True:
        try:
            if im.lower().endswith(('.mp4', '.avi', '.mov')):
                image = cv.VideoCapture(im)
                print("Video renders each frame individually, waiting for keystroke between frames. This is a slow process because of the calculations required for each image.")
                print("Hit ESC to quit.")
            else:
                image = cv.imread(im, cv.IMREAD_COLOR)
        except:
            print("Not a valid video or image.")
        else:
            print(image)
            print(im)
            break
else:
    import easygui
    im = easygui.fileopenbox(filetypes = ['*.jpg', '*.png', '*.mp4', '*.MP4', '*.avi', '*.AVI', '.mov'])
    if im.lower().endswith(('.mp4', '.avi', '.mov')):
        image = cv.VideoCapture(im)
        print("Video renders each frame individually, waiting for keystroke between frames. This is a slow process because of the calculations required for each image.")
        print("Hit ESC to quit.")
    else:
        image = cv.imread(im, cv.IMREAD_COLOR)
    #print(image)
    #cv.imshow("File", image)
    
finally:
    #loop if it's video, pausing between each frame for user to hit key
    if im.lower().endswith(('.mp4', '.avi', '.mov')):
        while(image.isOpened()):
            ret, frame = image.read()
            findCandy(frame)
            k = cv.waitKey(0)
            if k == 27:
                break
    #otherwise only do it once
    else:
        findCandy(image)
        cv.waitKey(0)


    
    
    cv.destroyAllWindows()