#@authors Cory Peterson and Timothy Wells
import sys

import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib import cm
from sklearn.cluster import MiniBatchKMeans
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
    (h, w) = image.shape[:2]
    image = cv.GaussianBlur(image,(31,31),0)
    image = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
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
    #cv.imshow("Image", image)
    #cv.show()
    print('yes')

    clt = MiniBatchKMeans(n_clusters= 32)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    # convert from L*a*b* to RGB
    quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
    image = cv.cvtColor(image, cv.COLOR_LAB2BGR)

    # display the images and wait for a keypress
    cv.imshow("image", np.hstack([image, quant]))
    cv.waitKey(0)


finally:
    #cv.imshow("File", image)
    cv.waitKey(0)
    cv.destroyAllWindows()