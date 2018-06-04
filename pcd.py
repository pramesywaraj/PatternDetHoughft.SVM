# import the necessary packages
import cv2
import os, os.path
import numpy as np
import glob

def hough_circle(file_loc):

    flag = TRUE

    valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]  # specify your vald extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    extension = os.path.splitext(file_loc)[1]

    if extension.lower() not in valid_image_extensions:
        return flag == FALSE
    else:

        img = cv2.imread(file_loc, 0)

        gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 300, param1=290, param2=55, minRadius=0, maxRadius=0)
        circles = np.uint16(np.around(circles))
        cropSize = (120, 120)

        if(str(circles) == "None"):
            return flag == FALSE

        else:
            for i in circles[0, :]:
                # cv2.circle(gray, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # cv2.circle(gray, (i[0], i[1]), 2, (0, 0, 255), 3)
                cropCoords = (
                max(0, i[1] - cropSize[0] // 2), min(img.shape[0], i[1] + cropSize[0] // 2), max(0, i[0] - cropSize[1] // 2),
                min(img.shape[1], i[0] + cropSize[1] // 2))
                crop = gray[cropCoords[0]:cropCoords[1], cropCoords[2]:cropCoords[3]]

    cv2.imwrite(file_loc, crop)
    return flag