import sys

import cv2 as cv
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import colors
#from matplotlib import cm
#from sklearn.cluster import MiniBatchKMeans

kernel = np.ones((5,5), np.uint8)
image = None

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
    #print("yeehaw")
    #image = cv.GaussianBlur(image,(11,11),0)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = cv.blur(image, (3,3))
    print("yeet")

    
 

    z = image.reshape((-1,3))
    z = np.float32(z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 16
    ret,label,center = cv.kmeans(z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    
    canny = cv.Canny(res2, 150, 300)
    
    '''circles = cv.HoughCircles(canny, cv.HOUGH_GRADIENT, 1.2, 10)
    # ensure at least some circles were found
    if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	    circles = np.round(circles[0, :]).astype("int")
    
	# loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        cv.circle(gray, (x, y), r, (0, 255, 0), 4)
        cv.rectangle(gray, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)'''

    img1, contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    
    cv.drawContours(res2, contours, -1, (0,255,0), 3)

    for i in contours:
        length = cv.arcLength(i, False)
        if length > 100:
            '''x,y,w,h = cv.boundingRect(i)
            cv.rectangle(res2, (x,y),(x+w,y+h),(0,0,255),4)'''
            (x,y),radius = cv.minEnclosingCircle(i)
            center = (int(x),int(y))
            radius = int(0.6*radius)
            if radius > 5 and radius < 50:
                cv.circle(res2,center,radius,(0,0,255),2)


    cv.imshow('Blurred & Color Reduced', res2)
    cv.imshow('Canny', canny)
    cv.imshow('Gray', gray)
    cv.waitKey(0)
    cv.destroyAllWindows()