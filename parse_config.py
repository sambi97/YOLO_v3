import json
import numpy as np
from PIL import Image

# Reading yolov3 configuration file
def parse_yolo_config(config_file):
	f = open(config_file,'r')
	lines = f.read().split('\n')
	lines = [i for i in lines if len(i) > 0]
	lines = [i.rstrip().lstrip() for i in lines if i[0] != "#"]

	layer = {}
	net = []

	for line in lines:
		if line[0] == "[":
			if len(layer) != 0:
				net.append(layer)
				layer = {}
			layer["type"] = line[1:-1].lstrip()
		else:
			temp1,temp2 = line.split("=")
			layer[temp1.rstrip()] = temp2.lstrip()
	net.append(layer)
	return net

# dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
# {'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
# {'segmentation': [[510.66, 423.01, 511.72, 420.03, 510.45, 416.0, 510.34, 413.02, 510.77, 410.26, 510.77, 407.5, 510.34, 405.16, 511.51, 402.83, 511.41, 400.49, 510.24, 398.16, 509.39, 397.31, 504.61, 399.22, 502.17, 399.64, 500.89, 401.66, 500.47, 402.08, 499.09, 401.87, 495.79, 401.98, 490.59, 401.77, 488.79, 401.77, 485.39, 398.58, 483.9, 397.31, 481.56, 396.35, 478.48, 395.93, 476.68, 396.03, 475.4, 396.77, 473.92, 398.79, 473.28, 399.96, 473.49, 401.87, 474.56, 403.47, 473.07, 405.59, 473.39, 407.71, 476.68, 409.41, 479.23, 409.73, 481.56, 410.69, 480.4, 411.85, 481.35, 414.93, 479.86, 418.65, 477.32, 420.03, 476.04, 422.58, 479.02, 422.58, 480.29, 423.01, 483.79, 419.93, 486.66, 416.21, 490.06, 415.57, 492.18, 416.85, 491.65, 420.24, 492.82, 422.9, 493.56, 424.39, 496.43, 424.6, 498.02, 423.01, 498.13, 421.31, 497.07, 420.03, 497.07, 415.15, 496.33, 414.51, 501.1, 411.96, 502.06, 411.32, 503.02, 415.04, 503.33, 418.12, 501.1, 420.24, 498.98, 421.63, 500.47, 424.39, 505.03, 423.32, 506.2, 421.31, 507.69, 419.5, 506.31, 423.32, 510.03, 423.01, 510.45, 423.01]], 'area': 702.1057499999998, 'iscrowd': 0, 'image_id': 289343, 'bbox': [473.07, 395.93, 38.65, 28.67], 'category_id': 18, 'id': 1768}
# {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}

def parse_data_config(config_file,ff):
	names = {}
	img_paths = {}
	with open(config_file) as f:
		j = json.load(f)

	for i in j["categories"]:
		names[i["id"]] = i["name"]

	for i in j["images"]:
		img_paths[i["id"]] = {"name": i["file_name"]}

	for i in j["annotations"]:
		if "bbox" not in img_paths[i["image_id"]]:
			img_paths[i["image_id"]]["bbox"] = [[i["category_id"]] + i["bbox"]]
		else:
			img_paths[i["image_id"]]["bbox"].append([i["category_id"]] + i["bbox"])

	temp = dict(img_paths)
	for i in img_paths:
		if  "bbox" not in img_paths[i]:
			del temp[i]

	with open(ff+".json", 'w') as f:
		json.dump({"names":names, "imgs": temp},f)
	#return names, img_paths

def check(file1, file2, pp):
	print ("checking")
	print (file1, file2, pp)
	img_files = open(file1,'r').readlines()
	img_files = [path[1:] for path in img_files]
	label_files = [path.replace(pp, file2).replace('.png','.txt').replace('.jpg','.txt') for path in img_files]
	print (label_files[0])
	indices = []

	for index in range(0, len(img_files)):
		img_path = img_files[index % len(img_files)].rstrip()
		label_path = label_files[index % len(img_files)].rstrip()
		if index % 5000 == 0:
			print(index)
		try:
			label = np.loadtxt(label_path)
			img = np.array(Image.open(img_path))
			if len(img.shape) < 3:
				indices.append(index)
		except:
			indices.append(index)

	for i in range(len(indices)):
		del img_files[indices[i] - i]
		del label_files[indices[i] - i]

	# print (len(img_files), len(label_files))
	with open("train_2014.txt", 'w') as f:
		f.write("".join(img_files))
	# return img_files, label_files

if __name__ == "__main__":
	# check("data/5k.txt", "data/labels/val2014", "images/val2014")
	check("data/trainvalno5k.txt", "data/labels/train2014", "images/train2014")