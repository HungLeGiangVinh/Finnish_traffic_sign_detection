import cv2 as cv
import os
import numpy as np
import argparse
from collections import deque
import pandas as pd

global csv, fname, legend, imgs
imgs = []

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes are integers, convert them to floats --
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	# initialize the list of picked indexes
	pick = []

	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 0]
	y1 = boxes[:, 1]
	x2 = boxes[:, 2]
	y2 = boxes[:, 3]
	scr = boxes[:, 5]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(scr)

	# keep looping while some indexes still remain in the indexes list
	while len(idxs) > 0:
		# grab the last index in the indexes list
		# and add the index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
											   np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# float data type
	return boxes[pick].astype("float")

# load csv entries in a similar format to csv files created during record file generation
def csv_maker(imgs, file, xsize, ysize, label, xmin, ymin, xmax, ymax):
	value = (file,
			xsize,
			ysize,
			label,
			xmin,
			ymin,
			xmax,
			ymax
			)
	imgs.append(value)
	column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
	csv_mk = pd.DataFrame(imgs, columns = column_name)
	return csv_mk

# combines image and legend
def result_creator(img, legend):
	h1, w1 = img.shape[:2]
	h2, w2 = legend.shape[:2]
	# create empty matrix
	vis = np.zeros((max(h1, h2), w1+w2, 3), np.uint8)
	# combine the image with the legend
	vis[:h1, :w1, :3] = img
	vis[:h2, w1:w1+w2, :3] = legend
	return vis

# create the legend
def legend_creator(colors, classes, height):
	# initialize the legend visualization
	legend = np.zeros((height, 300, 3), dtype="uint8")
	# loop over the class names + colors
	for (i, (className, color)) in enumerate(zip(classes, colors)):
		# draw the class name + color on the legend
		color = [int(c) for c in color]
		cv.rectangle(legend, (0, int(i * 25)), (int(300), int((i * 25 + 25) )), tuple(color), -1)
		cv.putText(legend, className, (int(5), int((i * 25 + 17))), cv.FONT_HERSHEY_SIMPLEX, 0.5 * 0.8, (0, 0, 0), 1, cv.LINE_AA)
	return legend

# object detection and processing function
def image_processing(cvNet,colors, classes):
	file_count = 0
	processed_count = 1
	csv = pd.DataFrame()
	print("Getting the number of files to processed ...")
	for _, _, files in os.walk(args.images):
		for file in files:
			if file.endswith(("jpg","jpeg","png","bmp")):
				file_count += 1
	print("There are {} image files in the target folder.".format(file_count))
	#perform object detection on all appropriate images in the folder.
	for root, _, files in os.walk(args.images):
		for file in files:
			# openCV will load these types of images and perform object detection on them
			if file.endswith(("jpg","jpeg","png","bmp")):
				print("Processing {}".format(file))
				print("{}/{} files in folder {}".format(processed_count,file_count,root))
				img = cv.imread(os.path.join(root, file))

				# image dimensions
				height = img.shape[0]
				width = img.shape[1]

				legend = legend_creator(colors,classes, height)
				# set image as input for the model, resize input to increase processing speed is optional but remember that accuracy will decrease
				cvNet.setInput(cv.dnn.blobFromImage(img, swapRB=True, crop=False))
				# output of model's prediction of objects in image
				cvOut = cvNet.forward()

				boxes = deque()
				boundingBoxes = []

				# load the predictions and filter out bounding boxes with low certainty(score)
				# structure of 'prediction': 
				# [<unknown(always 0)>, <label number(correspond to label map), <certainty(in percentage)>,
				# <x0 of the upper left point of the prediction box>, <y0 of the upper left point of the prediction box>, 
				# <x1 of the lower right point of the prediction box>, <y1 of the lower right point of the prediction box>]

				# all coordinates are in percentage will need to be multiply with the image row and column size to get a precise number
				for prediction in cvOut[0, 0, :, :]:
					score = float(prediction[2])
					if score > 0.5:
						label = prediction[1]
						startX = prediction[3] * width
						startY = prediction[4] * height
						endX = prediction[5] * width
						endY = prediction[6] * height
						boxes.append((startX, startY, endX, endY, label, score*100))
						boundingBoxes = np.array(boxes)

				# perform non-maximum suppression on the bounding boxes
				# boxes with too much overlap with another will be filter out based on their score
				pick = non_max_suppression_fast(boundingBoxes, 0.5)
				# loop over the picked bounding boxes and draw them
				for (startX, startY, endX, endY, label, score) in pick:
					cv.rectangle(img, (int(startX), int(startY)), (int(endX), int(endY)), colors[int(label)], thickness=2)
					textSize = cv.getTextSize('{:.2f}'.format(score), cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
					cv.rectangle(img, (int(startX), int(startY - textSize[0][1])), (int(startX + textSize[0][0]), int(startY)), colors[int(label)], thickness=-1)
					cv.putText(img, '{:.2f}'.format(score), (int(startX), int(startY)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
					if args.output_csv != None:
						csv = csv_maker(imgs, file, width, height, classes[int(label)], int(startX), int(startY), int(endX), int(endY))
				
				# implements human face blurring. Simple but unreliable, high failure rate.
				# draw bounding boxes to annotate blurred region
				if args.blur == True:
					gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
					# XML files are Haar Cassacades used for object detection
					# detect faces looking forward and apply Gaussian blur on them
					face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
					faces = face_cascade.detectMultiScale(gray, 1.3, 5)
					
					for (x, y, w, h) in faces:
						cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
						sub_face = img[y:y+h, x:x+w]
						sub_face = cv.GaussianBlur(sub_face, (23, 23), 30)
						img[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
					
					# detect human face profile and apply Gaussian blur on them
					profile_cascade = cv.CascadeClassifier('haarcascade_profileface.xml')
					profiles = profile_cascade.detectMultiScale(gray,1.3,5)
					for (x, y, w, h) in profiles:
						cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
						sub_face = img[y:y+h, x:x+w]
						sub_face = cv.GaussianBlur(sub_face, (23, 23), 30)
						img[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
				
				vis = result_creator(img, legend)
				if args.output_image_dir != None:
					if not os.path.exists(args.output_image_dir):
						os.makedirs(args.output_image_dir)
					cv.imwrite(os.path.join(args.output_image_dir, file),vis)
				
				# runs if view_result is true, stop image display and processing by pressing 'Escape' on the keyboard
				if args.view_result == True:
					cv.namedWindow(file,cv.WINDOW_KEEPRATIO)
					cv.moveWindow(file, 20,0)
					cv.imshow(file, vis)
					key = cv.waitKey(0)
					cv.destroyAllWindows()
					if key == 27:
						return
				processed_count += 1

	if args.output_csv != None and csv.empty != True:
		csv.to_csv(args.output_csv, index=None)
		print("Successfully created csv file at {}".format(os.path.join(os.getcwd(),args.output_csv)))
	elif csv.empty != True:
		print("No entries can be added to csv file! No csv file created.")
	return

def main():
	root_dir = os.path.dirname(os.path.abspath(__file__))
	print("Starting...")
	if args.output_csv != None:
		try:
			with open(args.output_csv, 'x') as tempfile:
				pass
		except OSError:
			print("Path to csv file is invalid!")
			quit()

	# default path values
	if args.graph == None:
		args.graph = os.path.join(root_dir, 'trained_model', 'frozen_inference_graph.pb')
	if args.graph_pbtxt == None:
		args.graph_pbtxt = os.path.join(root_dir, 'trained_model','graph_new.pbtxt')
	if args.images == None:
		args.images = os.path.join(root_dir,'images','demo')
	if args.labels_file == None:
		args.labels_file = os.path.join(root_dir, 'predefined_classes.txt')

	# load the predefined labels
	with open(args.labels_file, 'r') as f:
		classes = f.read().splitlines()

	# color list for each class (BGR format)
	colors = [[255, 255, 255],
			  [255, 128, 0],
			  [255, 255, 0],
			  [128, 255, 0],
			  [0, 255, 0],
			  [0, 255, 128],
			  [0, 255, 255],
			  [0, 128, 255],
			  [0, 0, 255]]

	# load the model using OpenCV' DNN module which includes a pb file and a pbtxt file
	# for more information on what models have been tested for OpenCV, look here: https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
	cvNet = cv.dnn.readNetFromTensorflow(args.graph,args.graph_pbtxt)

	image_processing(cvNet, colors, classes)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="This program takes a directory of .jpg images and perform object detection on them by importing our trained model \
		using OpenCV's library. The model is capable of detecting traffic signs. The user can also choose to detect human faces and blur them. Another option is \
		whether to choose to save the result as file or not. WARNING: Results for each time the program runs is not deterministic. The detected traffic \
		signs may be slightly different for different runs.")
	parser.add_argument("-g",
						"--graph",
						help="Input pb file of the model. Defaults to '/trained_model/frozen_inference_graph.pb' if none specified",
						type=str)
	parser.add_argument("-t",
						"--graph_pbtxt",
						help="Input pbtxt file of the model. Defaults to '/trained_model/graph.pbtxt' if none specified",
						type=str)
	parser.add_argument("-i",
						"--images",
						help="Path to images for the model to process. Defaults to '/images/' if none specified",
						type=str)
	parser.add_argument("-l",
						"--labels_file",
						help="Path to and name of .txt file containing the labels. If not specified, default to \
							'predefined_classes.txt' in current directory",
						 type=str)
	parser.add_argument("-b",
						"--blur",
						help="Option to blur human faces. Unreliable, may create false positives.",
						type=bool,
						default=True)
	parser.add_argument("-v",
						"--view_result",
						help="Option to view result of predictions, will go through files one by one. Press 'Escape' to stop.",
						type=bool,
						default = False)
	parser.add_argument("-o",
						"--output_image_dir",
						help="Option to save image of result predictions to specified path. Note that saved results are image files \
							with the same name as the original. Any existing image with the same name in the output directory will be overwritten",
						type=str)
	parser.add_argument("-c",
						"--output_csv",
						help="Option to save result of predictions to csv file which is in a similar format to csv output of record file generation. See 'tfrecord_generator'",
						type=str)
	args = parser.parse_args()
	main()
