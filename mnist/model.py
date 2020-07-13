import torch

import torchvision 

import torch.nn as nn

# useful functions like activation functions and convolution operations to use 
# are arithmetical operations, not have trainable params: weights and bias terms 

import torch.nn.functional as F 

# define your models as subclasses of torch.nn.models

# __init__ - initialize the layers you want to use 
class CNN_model(nn.Module): # nn.Module is for the params: weights and bias
	def __init__(self):
		super(CNN_model, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=10,  kernel_size=5) # (N, Channel, Hin, Win); (N, Cout, Hout, Wout)
		#self.pool = nn.maxpool()
		self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)

		self.fc1 = nn.Linear(320, 128)
		self.fc2 = nn.Linear(128, 64)
		self.out = nn.Linear(64, 10)


	def forward(self, x): # each forward pass 
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		# view() - fast and memory efficient shaping 
		x = x.view(-1, 320)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.out(x)

		return F.log_softmax(x)
