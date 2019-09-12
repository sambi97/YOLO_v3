from parse_config import parse_yolo_config
from data import *
from utils import *
from darknet import *

import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms, utils
from pathlib import Path
import time

if __name__ == "__main__":
	train_path = "data/trainvalno5k.txt"
	valid_path = "data/5k.txt"

	model = Darknet("config/yolov3.cfg", 416).cuda()

	coco = Coco_Dataset(valid_path,"data/labels/val2014")
	dataloader = DataLoader(coco, batch_size = 8, shuffle = True, num_workers = 4)
	optimizer = torch.optim.Adam(model.parameters(), 0.001)
	min_loss = float('inf')

	print ("YOLOv3 started training --------------------")

	epp = []
	for epoch in range(1):
		start = time.time()
		for batch_idx, (_, imgs, targets) in enumerate(dataloader):
			imgs = imgs.type(torch.cuda.FloatTensor)
			targets = targets.type(torch.cuda.FloatTensor)
			optimizer.zero_grad()
			loss = model(imgs,targets)
			loss.backward()
			optimizer.step()
			# print('Epoch:{}, Batch:{}, x_loss:{:0.4f}, y_loss:{:.4f}, w_loss:{:.4f}, h_loss:{:.4f}, conf:{:.4f},cls:{:.4f}, precision:{:.4f},recall:{:.4f}, total:{:.4f}'\
   #              .format(epoch, batch_idx,\
   #              model.losses["x"],model.losses["y"],\
   #              model.losses["w"],model.losses["h"],\
   #              model.losses["conf"],model.losses["cls"],\
   #              model.losses["recall"],model.losses["precision"],\
   #              loss.item()))
			model.seen += imgs.size(0)

			if loss.item() < min_loss:
				print('Better model found, saving it...')
				for f in Path("weights").glob('*.weights'):
					f.unlink()
				min_loss = loss.item()
				model.save_model('{}/{:.4f}.weights'.format("weights",min_loss))
				print('Saved!')
		end = time.time()
		print ("Epoch: ", epoch, "Loss : ", loss.item(), "Min_Loss : ", min_loss)
		print ("Time Elapsed : ", end - start)