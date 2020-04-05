import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
import os
from collections import Counter 
import json
import torch.utils.data as d

train_path = "./aclImdb/train"
test_path = "./aclImdb/test"

def reader(path):
	pos_path = os.path.join(path, "pos")
	neg_path = os.path.join(path, "neg")
	data = []
	label = []
	for file in os.listdir(pos_path):
		f = open(os.path.join(pos_path, file))
		data.append(f.read())
		label.append(1)
	for file in os.listdir(neg_path):
		f = open(os.path.join(neg_path, file))
		data.append(f.read())
		label.append(0)
	# print(data[:1])
	return data, label

def build_vocab(path, min_word_count = 5):
	pos_path = os.path.join(path, "pos")
	neg_path = os.path.join(path, "neg")
	data = []
	label = []
	counter = Counter()
	for file in os.listdir(pos_path):
		f = open(os.path.join(pos_path, file))
		line = f.read()
		line = gensim.utils.simple_preprocess(line) 
		data.append(line)
		counter.update(line)

	for file in os.listdir(neg_path):
		f = open(os.path.join(neg_path, file))
		line = f.read()
		line = gensim.utils.simple_preprocess(line) 
		data.append(line)
		counter.update(line)

	word2id = {}
	word2id['<pad>'] = 0
	word2id['<unk>'] = 1
	# print(data[:1])
	words = [word for word, count in counter.items() if count>min_word_count]

	for i, word in enumerate(words):
		word2id[word] = i+2
	
	with open("word2id.json", 'w') as f:
		json.dump(word2id, f)
	return word2id

class Dataset(d.Dataset):
	def __init__(self, word2id, train_path):
		self.word2id = word2id
		self.train_path = train_path
		self.data, self.label = reader(train_path)

	def __getitem__(self, index):
		seq = self.preprocess(self.data[index])
		label = torch.Tensor(self.label[index])
		return seq, label

	def __len__(self):
		return(len(self.data))

	def preprocess(self, text):
		line = gensim.utils.simple_preprocess(text)
		seq = []
		for word in line:
			if word in self.word2id:
				seq.append(self.word2id[word])
			else:
				seq.append(self.word2id['<unk>'])
		seq = torch.Tensor(seq)
		return seq

def collate_fn(data):
	data.sort(key=lambda x: len(x[0]), reverse=True)
	sequences, label = zip(*data)
	length = [len(seq) for seq in sequences]
	padded_seq = torch.zeros(len(sequences), max(length)).long()
	for i, seq in enumerate(sequences):
		end = length[i]
		padded_seq[i,:end] = seq
	return padded_seq, length, label



def dataloader(word2id, train_path, test_path, batch_size = 100):
	train_dataset = Dataset(word2id, train_path)
	test_dataset = Dataset(word2id, test_path)
	train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
	test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	return train_dataloader, test_dataloader