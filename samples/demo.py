import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# import utils 
# reload(utils)


import utils
import model as modellib
import visualize


import cv2

def load_files():
	# Root directory of the project
	ROOT_DIR = os.path.abspath("../")

	# Import Mask RCNN
	sys.path.append(ROOT_DIR)  # To find local version of the library
	# Import COCO config
	sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
	import coco


	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	# Download COCO trained weights from Releases if needed
	if not os.path.exists(COCO_MODEL_PATH):
	    utils.download_trained_weights(COCO_MODEL_PATH)

	# Directory of images to run detection on
	IMAGE_DIR = os.path.join(ROOT_DIR, "images")

	class InferenceConfig(coco.CocoConfig):
	    # Set batch size to 1 since we'll be running inference on
	    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	    GPU_COUNT = 1
	    IMAGES_PER_GPU = 1

	config = InferenceConfig()
	# config.display()

	return MODEL_DIR, config, COCO_MODEL_PATH


def infer(MODEL_DIR, config, COCO_MODEL_PATH, image):
	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)

	# COCO Class names
	# Index of the class in the list is its ID. For example, to get ID of
	# the teddy bear class, use: class_names.index('teddy bear')
	class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
	               'bus', 'train', 'truck', 'boat', 'traffic light',
	               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
	               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
	               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
	               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	               'kite', 'baseball bat', 'baseball glove', 'skateboard',
	               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
	               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
	               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
	               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
	               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
	               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
	               'teddy bear', 'hair drier', 'toothbrush']


	# Run detection
	results = model.detect([image], verbose=0)

	# Visualize results
	r = results[0]

	return r, class_names


def print_info(r):
	print("\nRegion of interest\n")
	print(r['rois'], "\n")

	print("Class_ids\n")
	print(r["class_ids"], "\n")

	print("Masks\n")
	print(r['masks'], "\n")


def visulaize_result(r, image, class_names):
	visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
	                            class_names, r['scores'])

def display_image(image, window_name="image"):
	cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
	cv2.imshow(window_name, image)
	cv2.waitKey(0)


def get_masked_image(image, mask):
	h = mask.shape[0]
	w = mask.shape[1]
	
	for y in range(0, h):
		for x in range(0, w):
			if(mask[y, x] == False):
				image[y, x] = 0


def process_mask(result, image, class_names):
	ids = result['class_ids']
	masks = result['masks']
	total_instances = ids.shape[0]
	final_mask = np.zeros(shape=(480, 640), dtype=bool)

	for i in range(0, total_instances):
		if(ids[i] == class_names.index('chair')):
			mask = masks[:, :, i]
			final_mask += mask

	get_masked_image(image, final_mask)


def extract_image_name(path):
	final_name = ''
	
	for char in path:
		if(char.isdigit()):
			final_name += char

	return final_name + "_segmented.jpg"


def main():
	MODEL_DIR, config, COCO_MODEL_PATH = load_files()
	image = skimage.io.imread(sys.argv[1])
	masked_image = cv2.imread(sys.argv[1])
	final_image_name = extract_image_name(sys.argv[1])
	
	result, class_names = infer(MODEL_DIR, config, COCO_MODEL_PATH, image)
	# print_info(result)
	visulaize_result(result, image, class_names)
	
	process_mask(result, masked_image, class_names)
	display_image(masked_image)
	
	cv2.imwrite(final_image_name, masked_image)
	print(final_image_name)

if __name__ == '__main__':
	main()