import numpy as np
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from skimage.transform import resize
from parse_config import check

# Custom dataset for COCO data
class Coco_Dataset(Dataset):
	def __init__(self, train_path, label_path , img_size = 416):
		# setting root of images path
		self.img_files, self.label_files = open(train_path,'r').readlines() #check(train_path, label_path, "images/val2014") 
		self.label_files = [path.replace("images/val2014", label_path).replace('.png','.txt').replace('.jpg','.txt') for path in self.img_files]
		print (len(self.img_files), len(self.label_files))
		self.img_shape = img_size
		self.max_objects = 50

	def __getitem__(self, index):
		# preparing image path
		img_path = self.img_files[index % len(self.img_files)].rstrip()

		img = np.array(Image.open(img_path))
		label_path = self.label_files[index % len(self.img_files)].rstrip()

		h, w, _ = img.shape
		dim_diff = np.abs(h-w)
		pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
		pad = ((pad1,pad2),(0,0),(0,0)) if h <= w else ((0,0),(pad1,pad2),(0,0))
		input_img = np.pad(img, pad, 'constant', constant_values=128)/255.0
		padded_h, padded_w, _ = input_img.shape
		input_img = resize(input_img, (self.img_shape, self.img_shape, 3), mode='reflect',anti_aliasing=True)
		input_img = np.transpose(input_img, (2,0,1))
		input_img = torch.from_numpy(input_img).float()

		labels = np.loadtxt(label_path).reshape(-1,5)
		x1 = w * (labels[:,1] - labels[:,3]/2)
		y1 = h * (labels[:,2] - labels[:,4]/2)
		x2 = w * (labels[:,1] + labels[:,3]/2)
		y2 = h * (labels[:,2] + labels[:,4]/2) 

		x1 += pad[1][0]
		y1 += pad[0][0]
		x2 += pad[1][0]
		y2 += pad[0][0]

		labels[:,1] = ((x1+x2)/2)/padded_w
		labels[:,2] = ((y1+y2)/2)/padded_h
		labels[:,3] *= w/padded_w
		labels[:,4] *= h/padded_h

		filled_labels = np.zeros((self.max_objects, 5))
		if labels is not None:
			filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
		filled_labels = torch.from_numpy(filled_labels)
		# print (input_img.shape, filled_labels.shape)
		return img_path, input_img, filled_labels

	def __len__(self):
		return len(self.img_files)

if __name__ == "__main__":
	cc = Coco_Dataset("E:", "data/5k.txt", "data/labels/val2014")
	d = DataLoader(cc)
	count = 0
	for i in d:
		break