import numpy as np
import torch
from torch.autograd import Variable
import cv2

def get_test():
	img = cv2.imread("1.png")
	img = cv2.resize(img, (416,416))
	img1 = img[:,:,::-1].transpose((2,0,1))
	img1 = img1[np.newaxis,:,:,:]/255.0
	img1 = torch.from_numpy(img1).float()
	img1 = Variable(img1)
	return img1