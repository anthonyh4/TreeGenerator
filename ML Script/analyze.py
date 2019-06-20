import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as torchmodels
import torch.utils.data.dataloader as dataloader
import torch.utils.data.dataset as dataset
import json
import os
import numpy as np
import operator
from PIL import Image
import argparse
from tqdm import tqdm
import copy
import math

CENTER_CROP_SIZE = 430
### Enumerate the parameters
# PARAMETER_NAMES = ['baseSplits', 'minRadius', 'scale', 'levels', 'ratio', 'baseSize_s', 
# 				'branchDist', 'seed', 'ratioPower', 'baseSize', 'splitHeight', 'scaleV', 'attractOut1', 
# 				'attractOut2', 'attractOut3', 'attractOut4', 'splitAngle1', 'splitAngle2', 'splitAngle3', 
# 				'splitAngle4', 'downAngleV1', 'downAngleV2', 'downAngleV3', 'downAngleV4', 'splitAngleV1', 
# 				'splitAngleV2', 'splitAngleV3', 'splitAngleV4', 'curveV1', 'curveV2', 'curveV3', 'curveV4', 
# 				'curve1', 'curve2', 'curve3', 'curve4', 'segSplits1', 'segSplits2', 'segSplits3', 'segSplits4', 
# 				'length1', 'length2', 'length3', 'length4', 'lengthV1', 'lengthV2', 'lengthV3', 'lengthV4', 
# 				'branches1', 'branches2', 'branches3', 'branches4', 'downAngle1', 'downAngle2', 'downAngle3', 'downAngle4', 'attractUp1', 
# 				'attractUp2', 'attractUp3', 'attractUp4', 'curveRes1', 'curveRes2', 'curveRes3', 'curveRes4']
PARAMETER_NAMES = ['length1', 'length2', 'length3', 'length4', 'lengthV1', 'lengthV2', 'lengthV3', 'lengthV4', 
'segSplits1', 'segSplits2', 'segSplits3', 'segSplits4', 'attractOut1', 'attractOut2', 'attractOut3', 'attractOut4',
'baseSize', 'baseSize_s', 'pruneWidth', 'splitHeight', 'ratio']

class ParameterFromImage(nn.Module):
	def __init__(self):
		super().__init__()
		res = torchmodels.resnet34(pretrained=True)
		num_in_features = res.fc.in_features
		#self.layer1_resnet = nn.Sequential(*list(res.children())[:-1])
		# res.fc = nn.Linear(num_in_features, 9)
		res.fc = nn.Linear(num_in_features, 21)
		self.layer1_resnet = res
		#self.layer2_to_modifiers = nn.Linear(num_in_features, 9)

	def forward(self, X):
		#print(X.size())
		X = self.layer1_resnet(X)

		# Modifier parameters must be from 0 to 1
		X = torch.sigmoid(X)
		#X = self.layer2_to_modifiers(X)
		return X

class RenderedHumanDataset(dataset.Dataset):
	def __init__(self, folderpath, bounds=None):
		self.path = folderpath
		self.tsfm = transforms.Compose([
			transforms.CenterCrop(CENTER_CROP_SIZE),
			transforms.Resize((224,224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
			])

		# with open(os.path.join(self.path, 'used_parameters.json')) as param_file:
		#     self.annotation_raw = json.load(param_file)
		param_file = [os.path.abspath(os.path.join(self.path, fp)) for fp in sorted(os.listdir(self.path)) if fp.endswith('.txt')]
		self.annotation_raw = dict()
		self.maxWeights = dict()
		for par in param_file:
			with open(par) as f:
				for line in f:
					self.annotation_raw[par[par.index("Image"):-3]+"png"] = eval(line)
					curDict = self.annotation_raw[par[par.index("Image"):-3]+"png"]
					maximum = 0
					for key in list(curDict.keys()):
						if curDict[key] is True:
							# curDict[key]=1
							curDict.pop(key)
						elif curDict[key] is False:
							# curDict[key]=0
							curDict.pop(key)
						elif type(curDict[key])==tuple:
							for i in range(len(curDict[key])):
								curDict[key+str(i+1)] = curDict[key][i]
								##update maximum
								# if curDict[key][i]>maximum:
								# 	maximum = curDict[key][i]
							curDict.pop(key)
						elif type(curDict[key])==list:
							for i in range(len(curDict[key])):
								curDict[key+str(i+1)] = curDict[key][i]
								##update maximum
								# if curDict[key][i]>maximum:
								# 	maximum = curDict[key][i]
							curDict.pop(key)
						else:
							##update maximum
							if curDict[key]>maximum:
								maximum = curDict[key]
					# for key in list(curDict.keys()):
					# 	curDict[key] = curDict[key]/maximum
					self.maxWeights[par[par.index("Image"):-3]+"png"] = maximum 
		self.filelist = [os.path.abspath(os.path.join(self.path, fp)) for fp in sorted(os.listdir(self.path)) if fp.endswith('.png')]
		if bounds is not None:
			low, high = bounds
			self.filelist = self.filelist[low:high]
			
		# print(map(operator.itemgetter(1), sorted(self.annotation_raw["Image0.png"].items(), key=operator.itemgetter(0))))
		# print(np.array(list(map(operator.itemgetter(1), sorted(self.annotation_raw["Image0.png"].items(), key=operator.itemgetter(0)))), dtype=np.float32))
		# self.modifiers = [np.array(list(map(operator.itemgetter(1), sorted(self.annotation_raw[imfile].items(), key=operator.itemgetter(0))))) for imfile in [os.path.relpath(fn,self.path) for fn in self.filelist]]

		self.modifiers = [np.array(list(map(operator.itemgetter(1), sorted(self.annotation_raw[imfile].items(), key=operator.itemgetter(0)))), dtype=np.float32) for imfile in [os.path.relpath(fn,self.path) for fn in self.filelist]]

	def __len__(self):
		return len(self.filelist)

	def __getitem__(self, i):
		fp, labels = self.filelist[i], self.modifiers[i]
		with Image.open(fp) as pilim:
			pilim = pilim.convert('RGB')
			im = self.tsfm(pilim)
		# self.filelist[i]
		return im, labels, os.path.basename(fp)
	
	@staticmethod
	def tensorToParameterDict(t):
		cp = t.cpu().numpy()
		keys = PARAMETER_NAMES
		print(cp.keys())
		assert len(keys) == len(cp),"Unexpected tensor result"
		params = {keys[i]: float(cp[i]) for i in range(len(keys))}
		del cp
		return params

	@staticmethod 
	def multitensorToParameterDict(t, fps):
		cp = t.cpu().numpy()
		keys = PARAMETER_NAMES
		params = {fps[j]: {keys[i]: float(cp[j][i]) for i in range(len(keys))} for j in range(len(fps))}
		del cp
		return params


def main():

	parser = argparse.ArgumentParser(description="Trains a NN to determine parameters from an image")
	
	parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default 1e-3)')
	parser.add_argument('--epochs', type=int, default=6, help='number of epochs (default 20)')
	parser.add_argument('--batch', type=int, default=16, help='batch size (default 16)')
	parser.add_argument('--valsize', type=float, default=0.1, help='fraction of data to use for validation (default 0.1)')
	parser.add_argument('--model', type=os.path.abspath, default=None, help='Starting model weights (default None)')
	parser.add_argument('--data', type=os.path.abspath, default='./renders', help='Directory containing renders (default ./renders)')
	# parser.add_argument('--eval', type=os.path.abspath, default=None, help='Instead of training, output classifications to this file.')
	parser.add_argument('--eval', type=os.path.abspath, default='./predictions/predict.json', help='Instead of training, output classifications to this file.')
	parser.add_argument('--ngpus', type=int, default=1, help="How many GPUs to train on (default 1)")
	parser.add_argument('--gpu', type=int, default=0, help="Which GPU index to use first (default 0)")
	args = parser.parse_args()
	# print(torch.cuda.is_available()) 
	# print(torch.cuda.get_device_capability(device=None))
	device = torch.device("cpu") # Uncomment this to run on GPU

	# num_examples = len(os.listdir(args.data)) - 1
	num_examples = int(len(os.listdir(args.data))/2)

	assert num_examples > 2, 'Need at least 2 renders in directory'

	num_val = math.ceil(args.valsize * num_examples)
	num_train = num_examples - num_val
	print('Splitting Train/Val: %d/%d' % (num_train, num_val))
	train_dataset = RenderedHumanDataset(args.data,bounds=(0,num_train))
	val_dataset = RenderedHumanDataset(args.data, bounds=(num_train,num_train+num_val))
	model = ParameterFromImage()
	# if torch.cuda.device_count() >= 1:
	# 	print("Using %d of %d GPUs!" % (args.ngpus, torch.cuda.device_count()))
	# 	model = torch.nn.DataParallel(model, range(args.gpu, min(torch.cuda.device_count(),args.gpu+args.ngpus)))
	# if args.model is not None and os.path.isfile(args.model):
	# 	print('Resuming from %s' % args.model)
	# 	model_weights = torch.load(args.model, map_location={'cuda:2':'cuda:0'})
	# 	model.load_state_dict(model_weights)
	model = model.to(device)

	lr = args.lr
	batch_size = args.batch
	n_epochs = args.epochs
	loss_fn = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	losses = {
		'train': [],
		'val': []
	}

	best_loss = math.inf
	best_model = None

	train_loader = dataloader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = dataloader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
	dataloaders = {"train": train_loader, "val": val_loader}
	
	phases_to_use = ['val'] if args.eval else ['train','val']
	if args.eval:
		n_epochs = 1
		print("Outputting predictions to %s" % str(args.eval))
	model_evalpredictions = {}
	for epoch in range(n_epochs):
		print("EPOCH %d:" % epoch)

		for phase in phases_to_use:
			print("%s:" % phase.title())
			if phase == 'train':
				model.train()
			else:
				model.eval()
			running_loss = 0.0
			for (batch_num, (im, labels, im_fnames)) in enumerate(tqdm(dataloaders[phase])):
				im = im.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()

				with torch.set_grad_enabled(phase == 'train'):
					predicted_modifiers = model(im)
					# print(im)
					# print(predicted_modifiers)

					loss = loss_fn(predicted_modifiers, labels)
					if(args.eval):
						model_evalpredictions.update(RenderedHumanDataset.multitensorToParameterDict(predicted_modifiers, im_fnames))
					if phase == 'train':
						loss.backward()
						optimizer.step()
				
				# Loss * batch size
				running_loss += loss.item() * im.size(0)

			epoch_loss = running_loss / len(dataloaders[phase].dataset)
			losses[phase].append(epoch_loss)
			if phase == 'val':
				if epoch_loss < best_loss:
					best_loss = epoch_loss
					best_model = copy.deepcopy(model.state_dict())
			print('Loss: %.4f' % (epoch_loss))
		print('\n')
		
	torch.save(losses, 'models/losses.pt')
	torch.save(best_model, 'models/model_best.pt')
	if(args.eval):
		with open(args.eval,'w') as f:
			json.dump(model_evalpredictions, f, indent=4)
	


if __name__ == "__main__":
	main()