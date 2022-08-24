#ussage python3 main.py --firstimage first.jpg --secondimage nn.png

import numpy as np
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--firstimage", required=True,
	help="path to the first image")
ap.add_argument("-r", "--secondimage", required=True,
	help="path to second image")
args = vars(ap.parse_args())


image2 = cv2.imread(args["firstimage"])
image = cv2.imread(args["secondimage"])

image = imutils.resize(image, width=740,height=313)
image2 = imutils.resize(image2, width=740,height=313)

cv2.imshow('d1 ',image)

#first image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)


# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure that at least one contour was found
if len(cnts) > 0:
	# sort the contours according to their size in
	# descending order
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	# loop over the sorted contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points,
		# then we can assume we have found the paper
		if len(approx) == 4:
			docCnt = approx
			break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper


paper = four_point_transform(image, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))



cv2.imshow('d111 ',paper)


#second image

gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
blurred2 = cv2.GaussianBlur(gray2, (5, 5), 0)
edged2 = cv2.Canny(blurred2, 75, 200)
cv2.imshow('d3 ',edged2)



# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts2 = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts2 = imutils.grab_contours(cnts2)
docCnt = None

# ensure that at least one contour was found
if len(cnts2) > 0:
	# sort the contours according to their size in
	# descending order
	cnts2 = sorted(cnts2, key=cv2.contourArea, reverse=True)

	# loop over the sorted contours
	for c in cnts2:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)

		# if our approximated contour has four points,
		# then we can assume we have found the paper
		if len(approx) == 4:
			docCnt = approx
			break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper2 = four_point_transform(image2, docCnt.reshape(4, 2))
warped2 = four_point_transform(gray2, docCnt.reshape(4, 2))

#cv2.imshow('paper2', paper2)


scale_percent = 40  # percent of original size
width = int(warped2.shape[1] * scale_percent / 20)
height = int(warped2.shape[0] * scale_percent / 20)
dim = (width, height)

# resize image
resized1 = cv2.resize(warped2, dim, interpolation=cv2.INTER_AREA)

resized2 = cv2.resize(warped, dim, interpolation=cv2.INTER_AREA)

img3 = cv2.subtract(resized2,resized1)
cv2.imshow('last',img3)

cv2.waitKey(0)
