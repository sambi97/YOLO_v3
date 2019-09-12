import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.autograd import Variable
import cv2
from collections import defaultdict

from parse_config import *
from utils import *

# Custom layers for yolov3
class EmptyLayer(nn.Module):
	def __init__(self):
		super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module):
	def __init__(self, anchors, num_classes, img_size=416, thresh=None):
		super(DetectionLayer, self).__init__()
		self.anchors = anchors
		self.num_classes = num_classes
		self.img_size = img_size
		self.mse_loss = nn.MSELoss(reduction='elementwise_mean')
		self.ce_loss = nn.CrossEntropyLoss() 
		self.lambda_coord = 5
		self.lambda_noobj = 0.5

	def forward(self, prediction, targets = None):
		batch_size = prediction.size(0)
		grid_size = prediction.size(2)
		stride = self.img_size // grid_size
		bbox_attrs = 5 + self.num_classes
		num_anchors = len(self.anchors)

		prediction = prediction.view(batch_size, num_anchors, bbox_attrs, grid_size, grid_size).permute(0, 1, 3, 4, 2).contiguous()
		scaled_anchors = torch.FloatTensor([(a[0]/stride, a[1]/stride) for a in self.anchors])

		x = torch.sigmoid(prediction[..., 0])
		y = torch.sigmoid(prediction[..., 1])
		w = prediction[..., 2]
		h = prediction[..., 3]
		pred_conf = torch.sigmoid(prediction[..., 4])
		pred_cls = torch.sigmoid(prediction[..., 5:])

		grid_x = torch.arange(grid_size).repeat(grid_size, 1).view([1, 1, grid_size, grid_size]).type(torch.cuda.FloatTensor)
		grid_y = torch.arange(grid_size).repeat(grid_size, 1).t().view([1, 1, grid_size, grid_size]).type(torch.cuda.FloatTensor)

		anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1)).cuda()
		anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1)).cuda()

		# print (type(anchor_w))
		pred_boxes = torch.cuda.FloatTensor(prediction[..., :4].shape)
		pred_boxes[..., 0] = x.data + grid_x
		pred_boxes[..., 1] = y.data + grid_y
		pred_boxes[..., 2] = torch.exp(w) * anchor_w
		pred_boxes[..., 3] = torch.exp(h) * anchor_h

		if targets is not None:
			if x.is_cuda:
				self.mse_loss = self.mse_loss.cuda()
				self.ce_loss = self.ce_loss.cuda()

			nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = buildTargets(
				pred_boxes=pred_boxes.cpu().data,
				pred_conf=pred_conf.cpu().data,
				pred_cls=pred_cls.cpu().data,
				targets=targets.cpu().data,
				anchors=scaled_anchors.cpu().data,
				num_anchors=num_anchors,
				num_classes=self.num_classes,
				grid_size=grid_size,
				ignore_thres=0.5,
				img_size=self.img_size)

			nProposals = int((pred_conf > 0.5).sum().item())
			recall = float(nCorrect / nGT) if nGT else 1
			precision = float(nCorrect / nProposals)

			mask = Variable(mask.type(torch.cuda.ByteTensor))
			conf_mask = Variable(conf_mask.type(torch.cuda.ByteTensor))
			tx = Variable(tx.type(torch.cuda.FloatTensor), requires_grad=False)
			ty = Variable(ty.type(torch.cuda.FloatTensor), requires_grad=False)
			tw = Variable(tw.type(torch.cuda.FloatTensor), requires_grad=False)
			th = Variable(th.type(torch.cuda.FloatTensor), requires_grad=False)
			tconf = Variable(tconf.type(torch.cuda.FloatTensor), requires_grad=False)
			tcls = Variable(tcls.type(torch.cuda.LongTensor), requires_grad=False)
			
			conf_mask_true = mask
			conf_mask_false = conf_mask - mask
			loss_x = self.lambda_coord * self.mse_loss(x[mask], tx[mask])
			loss_y = self.lambda_coord * self.mse_loss(y[mask], ty[mask])
			loss_w = self.lambda_coord * self.mse_loss(w[mask], tw[mask])
			loss_h = self.lambda_coord * self.mse_loss(h[mask], th[mask])

			loss_conf = self.lambda_noobj * self.mse_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.mse_loss(pred_conf[conf_mask_true], tconf[conf_mask_true])
			loss_cls = (1 / batch_size) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
			loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

			return [loss, loss_x.item(), loss_y.item(), loss_w.item(), loss_h.item(), loss_conf.item(), loss_cls.item(), recall, precision]
		
		else:
			prediction = torch.cat((pred_boxes.view(batch_size, -1, 4) * stride,pred_conf.view(batch_size, -1, 1),pred_cls.view(batch_size, -1, self.num_classes)),-1)
			prediction = prediction.view(batch_size, num_anchors, grid_size, grid_size, bbox_attrs).permute(0, 2,3,1,4).contiguous().view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
			return prediction


# Modules for darknet
def create_net(net, img_size):
	net_params = net[0]
	net_list = nn.ModuleList()

	index = 0
	prev_filters = int(net_params["channels"]) #img_depth
	output_filters = []

	for layer in net[1:]:
		module = nn.Sequential()

		if layer["type"] == "convolutional":
			
			if "batch_normalize" in layer:
				batch_normalize = layer["batch_normalize"]
				bias = False
			else:
				batch_normalize = 0
				bias = True

			filters = int(layer["filters"])
			kernel_size = int(layer["size"]) #filter
			pad = (kernel_size - 1) // 2 if int(layer["pad"]) else 0 #padding
			stride = int(layer["stride"])

			conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
			module.add_module("conv_{0}".format(index), conv)

			if batch_normalize:
				module.add_module("batch_norm_{0}".format(index), nn.BatchNorm2d(filters, momentum=0.1))

			if  layer["activation"] == "leaky":
				module.add_module("leaky_{0}".format(index), nn.LeakyReLU(0.1, inplace = True))

		elif layer["type"] == "upsample":
			upsample = nn.Upsample(scale_factor = int(layer["stride"]), mode = "nearest")
			module.add_module("upsample_{0}".format(index), upsample)

		elif layer["type"] == "route":
			layers = [int(i) for i in layer["layers"].split(",")]
			if len(layers) > 1:
				filters = output_filters[index + layers[0]] + output_filters[layers[1]]
			else:
				filters= output_filters[index + layers[0]]

			route = EmptyLayer()
			module.add_module("route_{0}".format(index), route)

		elif layer["type"] == "shortcut":
			from_ = int(layer["from"])
			shortcut = EmptyLayer()
			module.add_module("shortcut_{0}".format(index), shortcut)

		elif layer["type"] == "yolo":
			mask = [int(i) for i in layer["mask"].split(",")]
			anchors = [int(i) for i in layer["anchors"].split(",")]
			anchors = [(anchors[i], anchors[i+1]) for i in range(0,len(anchors),2)]
			anchors = [anchors[i] for i in mask]

			detection = DetectionLayer(anchors, int(layer['classes']),img_size)
			module.add_module("Detection_{0}".format(index), detection)

		else:
			print("Error")
		net_list.append(module)
		prev_filters = filters
		output_filters.append(filters)
		index += 1
	return (net_params, net_list)

class Darknet(nn.Module):
	def __init__(self, config_file, img_size):
		super(Darknet, self).__init__()
		self.net = parse_yolo_config(config_file)
		self.net_params, self.module_list = create_net(self.net, img_size)
		self.loss_names = ["x", "y", "w", "h", "conf", "cls", "recall", "precision"]
		self.seen = 0
		self.header = torch.IntTensor([0, 0, 0, self.seen, 0])

	def forward(self, x, y = None):
		modules = self.net[1:]
		self.losses = defaultdict(float)
		img_size = x.shape[2]
		outputs = {}
		yolo_out = []
		loss = 0

		for i in range(len(modules)):
			module_type = (modules[i]["type"])

			if module_type == "convolutional" or module_type == "upsample":
				x = self.module_list[i](x)

			elif module_type == "route":
				layers = modules[i]["layers"]
				layers = [int(a) for a in layers.split(",")]

				if len(layers) == 1:
					x = outputs[i+(layers[0])]
				else:
					x = torch.cat((outputs[i+layers[0]],outputs[layers[1]]),1)

			elif module_type == "shortcut":
				x = outputs[i-1] + outputs[i+int(modules[i]["from"])]

			elif module_type == "yolo":
				if y is not None:
					losses = self.module_list[i][0](x,y)
					x = losses[0]
					losses = losses[1:]
					for name, loss in zip(self.loss_names, losses):
						self.losses[name] += loss
				else:
					x = self.module_list[i][0](x,y)
				yolo_out.append(x)
			outputs[i] = x
			# outputs[i] = outputs[i-1]

		self.losses["recall"] /= 3
		self.losses["precision"] /= 3
		return sum(yolo_out) if y is not None else torch.cat(yolo_out, 1)

	def load_weights(self, weightfile):
		fp = open(weightfile, "rb")
		header = np.fromfile(fp, dtype = np.int32, count = 5)
		self.header = torch.from_numpy(header)
		self.seen = self.header[3]

		weights = np.fromfile(fp, dtype = np.float32)
		ptr = 0
		for i in range(len(self.module_list)):
			module_type = self.net[i + 1]["type"]

			if module_type == "convolutional":
				model = self.module_list[i]

				try:
					batch_normalize = int(self.net[i+1]["batch_normalize"])
				except:
					batch_normalize = 0
				conv = model[0]

				if (batch_normalize):
					bn = model[1]
					num_bn_biases = bn.bias.numel()

					bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
					ptr += num_bn_biases

					bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases

					bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases

					bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
					ptr  += num_bn_biases

					bn_biases = bn_biases.view_as(bn.bias.data)
					bn_weights = bn_weights.view_as(bn.weight.data)
					bn_running_mean = bn_running_mean.view_as(bn.running_mean)
					bn_running_var = bn_running_var.view_as(bn.running_var)

					bn.bias.data.copy_(bn_biases)
					bn.weight.data.copy_(bn_weights)
					bn.running_mean.copy_(bn_running_mean)
					bn.running_var.copy_(bn_running_var)

				else:

					num_biases = conv.bias.numel()
					conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
					ptr = ptr + num_biases
					conv_biases = conv_biases.view_as(conv.bias.data)
					conv.bias.data.copy_(conv_biases)

				num_weights = conv.weight.numel()
				conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
				ptr = ptr + num_weights
				conv_weights = conv_weights.view_as(conv.weight.data)
				conv.weight.data.copy_(conv_weights)

	def save_model(self, weightfile, cutoff = 0):
		fp = open(weightfile, 'wb')
		self.header[3] = self.seen
		header = self.header
		header = header.numpy()
		header.tofile(fp)

		for i in range(len(self.module_list)):
			module_type = self.net[i+1]["type"]
			if (module_type) == "convolutional":
				model = self.module_list[i]
				try:
					batch_normalize = int(self.net[i+1]["batch_normalize"])
				except:
					batch_normalize = 0 
				conv = model[0]
				if (batch_normalize):
					bn = model[1]
					bn.bias.data.cpu().numpy().tofile(fp)
					bn.weight.data.cpu().numpy().tofile(fp)
					bn.running_mean.cpu().numpy().tofile(fp)
					bn.running_var.cpu().numpy().tofile(fp)
				else:
					conv.bias.data.cpu().numpy().tofile(fp)
				conv.weight.data.cpu().numpy().tofile(fp)