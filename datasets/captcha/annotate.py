# USAGE
# python annotate.py --input downloads --annot dataset

# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
	help="path to output directory of annotations")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character
# counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# display an update to the user
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))

	try:
		# load the image and convert it to grayscale, then pad the
		# image to ensure digits caught only the border of the image
		# are retained
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8,
			cv2.BORDER_REPLICATE)

		# threshold the image to reveal the digits
		thresh = cv2.threshold(gray, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# detect the contours
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

		# loop over the contours
		for c in cnts:
			# 计算能宝威轮廓的最小的矩形bounding box
			(x, y, w, h) = cv2.boundingRect(c)
			roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

			# 显示图片，然后如果发现抠出来的不是数字，就按'`'去忽略这个切片
			cv2.imshow("ROI", imutils.resize(roi, width=28))
			key = cv2.waitKey(0)

			if key == ord("`"):
				print("[INFO] ignoring character")
				continue

			# 认为判断抠出来的是几，就按下哪个按键，程序自动将抠图保存在那个数字对应的文件夹下
			key = chr(key).upper()
			dirPath = os.path.sep.join([args["annot"], key])
			
			# if the output directory does not exist, create it
			if not os.path.exists(dirPath):
				os.makedirs(dirPath)

			# write the labeled character to file
			count = counts.get(key, 1)
			p = os.path.sep.join([dirPath, "{}.png".format(
				str(count).zfill(6))])
			cv2.imwrite(p, roi)
		
			# increment the count for the current key
			counts[key] = count + 1

	# we are trying to control-c out of the script, so break from the
	# loop (you still need to press a key for the active window to
	# trigger this)
	except KeyboardInterrupt:
		print("[INFO] manually leaving script")
		break

	# an unknown error has occurred for this particular image
	except:
		print("[INFO] skipping image...")