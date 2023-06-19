import torch
import torch.nn as nn


class MLPIC(nn.Module):
	""" 
	Name: MLPIC
	Version: 0.0.1
	Architecture: Multi-Layer Perceptron (MLP)

	Parameters:
	- n_vocab (int): Number of vocabulary
	- n_hidden (int): Number of hidden units
	- n_classes (int): Number of output classes
	"""
	def __init__(self, n_vocab: int, n_hidden: int, n_classes: int):
		super(MLPIC, self).__init__()
		self.l1 = nn.Linear(n_vocab, n_hidden)
		self.l2 = nn.Linear(n_hidden, n_hidden)
		self.l3 = nn.Linear(n_hidden, n_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		out = self.l1(x)
		out = self.relu(out)
		out = self.l2(out)
		out = self.relu(out)
		out = self.l3(out)
		return out