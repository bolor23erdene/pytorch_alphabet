import torch

import torchvision 

import torch.nn as nn

from model import CNN_model

import torch.optim as optim

nb_epochs = 2 

batch_size_train = 64

batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


cnn_model = CNN_model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)

for r in range(nb_epochs):

	total_loss = 0

	for i, batch in enumerate(train_loader, 0):

		images, labels = batch 

		optimizer.zero_grad()

		prediction = cnn_model(images) # softmax

		loss = criterion(prediction, labels)

		loss.backward() # compute the backpropagation 

		optimizer.step() # based on the computed gradients update the weights of the 

		total_loss += loss.item()

		if i % 100 == 0:
			print("Loss: ", total_loss/((i+1)*len(images)))









