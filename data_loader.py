import torch
import torch.utils.data as D

import os
import gensim

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

class Dataset(D.Dataset):
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
	train_dataloader = D.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
	test_dataloader = D.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	return train_dataloader, test_dataloader