#@authors Cory Peterson and Timothy Wells
import sys

import cv2 as cv
import numpy as np
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
    image = easygui.fileopenbox()
    print(image)
    #cv.imshow("File", image)
    image = cv.imread(image, cv.IMREAD_COLOR)
    print('yes')
finally:
    cv.imshow("File", image)
    cv.waitKey(0)
    cv.destroyAllWindows()