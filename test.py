from __future__ import division

import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import *
from data import *
from darknet import Darknet

if __name__ == "__main__":
	print ("Loading network weights --------------------")

	valid_path = "data/5k.txt"
	num_classes = 80

	model = Darknet("config/yolov3.cfg", 416).cuda()
	model.load_weights("weights/35.3313.weights")
	model.eval()
	FloatTensor = torch.cuda.FloatTensor

	coco = Coco_Dataset(valid_path,"data/labels/val2014")
	dataloader = DataLoader(coco, batch_size = 16, shuffle = True, num_workers = 4)

	all_detections = []
	all_annotations = []

	for batch_idx, (_, imgs, targets) in enumerate(dataloader):
		# print (batch_idx)
		imgs = Variable(imgs.type(FloatTensor))

		with torch.no_grad():
			outputs = model(imgs)
			outputs = nonMaxSuppression(outputs, 0.5, num_classes, True, 0.45).unsqueeze(0)

		for output, annotations in zip(outputs, targets):
			all_detections.append([np.array([]) for _ in range(num_classes)])
			if output is not None:
				pred_boxes = output[:, :5].cpu().numpy()
				scores = output[:, 4].cpu().numpy()
				pred_labels = output[:, -1].cpu().numpy()
				sort_i = np.argsort(scores)
				pred_labels = pred_labels[sort_i]
				pred_boxes = pred_boxes[sort_i]
				for label in range(num_classes):
					all_detections[-1][label] = pred_boxes[pred_labels == label]

			all_annotations.append([np.array([]) for _ in range(num_classes)])
			if any(annotations[:, -1] > 0):
				annotation_labels = annotations[annotations[:, -1] > 0, 0].numpy()
				_annotation_boxes = annotations[annotations[:, -1] > 0, 1:]
				annotation_boxes = np.empty_like(_annotation_boxes)
				annotation_boxes[:, 0] = _annotation_boxes[:, 0] - _annotation_boxes[:, 2] / 2
				annotation_boxes[:, 1] = _annotation_boxes[:, 1] - _annotation_boxes[:, 3] / 2
				annotation_boxes[:, 2] = _annotation_boxes[:, 0] + _annotation_boxes[:, 2] / 2
				annotation_boxes[:, 3] = _annotation_boxes[:, 1] + _annotation_boxes[:, 3] / 2
				annotation_boxes *= 416

				for label in range(num_classes):
					all_annotations[-1][label] = annotation_boxes[annotation_labels == label, :]

	average_precisions = {}
	for label in range(num_classes):
		true_positives = []
		scores = []
		num_annotations = 0
		for i in range(len(all_annotations)):
			detections = all_detections[i][label]
			annotations = all_annotations[i][label]

			num_annotations += annotations.shape[0]
			detected_annotations = []
			for *bbox, score in detections:
				print ("Dete: ",detections.shape, bbox.shape)
				scores.append(score)
				if annotations.shape[0] == 0:
					true_positives.append(0)
					continue
				overlaps = bboxIOUNumpy(np.expand_dims(bbox, axis=0), annotations)
				assigned_annotation = np.argmax(overlaps, axis=1)
				max_overlap = overlaps[0, assigned_annotation]

				if max_overlap >= 0.5 and assigned_annotation not in detected_annotations:
					true_positives.append(1)
					detected_annotations.append(assigned_annotation)
				else:
					true_positives.append(0)

		if num_annotations == 0:
			average_precisions[label] = 0
			continue

		true_positives = np.array(true_positives)
		false_positives = np.ones_like(true_positives) - true_positives
		indices = np.argsort(-np.array(scores))
		false_positives = false_positives[indices]
		true_positives = true_positives[indices]

		false_positives = np.cumsum(false_positives)
		true_positives = np.cumsum(true_positives)

		recall = true_positives / num_annotations
		precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

		average_precision = computeAP(recall, precision)
		average_precisions[label] = average_precision

	print("Average Precisions:")
	for c, ap in average_precisions.items():
		print(f"+ Class '{c}' - AP: {ap}")
	mAP = np.mean(list(average_precisions.values()))
	print(f"mAP: {mAP}")