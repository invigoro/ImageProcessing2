#@authors Cory Peterson and Timothy Wells
import sys

import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib import cm
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils
import argparse

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
    print(image)
    #cv.imshow("File", image)
    image = cv.imread(image, cv.IMREAD_COLOR)
    """    #Setting up color plot
    r, g, b = cv.split(image)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = image.reshape((np.shape(image)[0] * np.shape(image)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()
     #end plot"""
    print('yes')

    image = cv.GaussianBlur(image,(1,1),0)
    shifted = cv.pyrMeanShiftFiltering(image, 18, 56)

    gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
    thresh = cv.threshold(gray, 0, 255,
                           cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    cv.imshow("Thresh", thresh)

    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=25,
                              labels=thresh)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        # draw a circle enclosing the object
        ((x, y), r) = cv.minEnclosingCircle(c)
        cv.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show the output image
    cv.imshow("Output", image)
    cv.waitKey(0)
    cv.show()

finally:
    cv.imshow("File", image)
    cv.waitKey(0)
    cv.destroyAllWindows()