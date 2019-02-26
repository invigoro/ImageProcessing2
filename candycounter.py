import sys

import cv2 as cv
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import colors
#from matplotlib import cm
#from sklearn.cluster import MiniBatchKMeans

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
    image = cv.GaussianBlur(image,(11,11),0)
    z = image.reshape((-1,3))
    z = np.float32(z)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 16
    ret,label,center = cv.kmeans(z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((image.shape))

    cv.imshow('Blurred & Color Reduced', res2)
    cv.waitKey(0)
    cv.destroyAllWindows()
