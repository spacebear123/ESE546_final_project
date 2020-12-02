import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

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

if __name__ == "__main__":
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to trained weights file
    # Download this file and place in the root of your 
    # project (See README file for details)
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")

    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    model.load_weights(COCO_MODEL_PATH, by_name=True)
    # Load a random image from the images folder
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

    # Run detection
    results = model.detect([image], verbose=1)

    # Visualize results
    res = results[0]
    masks = res['masks']
    hz_masks = masks[:, :, mask_idxs]
    mask = np.logical_or.reduce(hz_masks, axis=2, keepdims=True)
    map_fn = np.vectorize(lambda x: 255 if x else 127)
    mask = map_fn(mask)
    catted = np.concatenate([image, mask.astype(np.int32)], axis=2)
    file_name_no_ext = os.path.splitext(file_name)[0]
    out_file_name = ".".join([file_name_no_ext, "npy"])
    scipy.misc.toimage(catted, cmin=0.0, cmax=255.).save('0.png')
