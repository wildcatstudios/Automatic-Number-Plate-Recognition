import torch
import os
import easyocr
import numpy as np
import cv2
import shutil

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', './best.pt')  # or yolov5m, yolov5l, yolov5x, custom
model.conf = 0.1
model.iou = 0.3

# Images
img = '/home/tim/Downloads/pexels-mike-van-schoonderwalt-5484515.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

save_pth = os.path.join(os.getcwd(), 'crop')

if os.path.exists(save_pth):
	shutil.rmtree(save_pth)

results.crop(save=True, save_dir="./crop")



save_pth = os.path.join(os.getcwd(), 'crop')
crop_dir = (os.path.join(save_pth, 'crops', 'licence_plate/'))
"""
if os.path.exists(crop_dir):
    for fname in os.listdir(crop_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            os.remove(save_pth)
sav = results.crop(save=True, save_dir=save_pth)
"""
def filter_txt(region, ocr_result, region_threshold):
    rect_size = region.shape[0]*region.shape[1]

    plate = []

    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        # print('res', result[1])
        # print('sol', length*height / rect_size)

        if length*height / rect_size >= region_threshold:
            plate.append(result[1])
            # print('rec size', length, height, rect_size)


    return plate
    


reader = easyocr.Reader(['en'])

if os.path.exists(crop_dir):
    for fname in os.listdir(crop_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        	# print(fname)
        	full = (crop_dir + fname)
        	# print(full)
        	rg = cv2.imread(full)
        	
        	result = reader.readtext(rg)
        	# print(result)
        	
        	txt = (filter_txt(rg, result, 0.4))
        	for i in txt:
        		print(''.join(i))
