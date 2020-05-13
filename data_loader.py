import torch
import torch.utils.data as D
import numpy as np
import os
import gensim

#simple function which read the data from directory and return data and label
# you can make your own reader for other dataset.
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

#function used for custom dataset in pytorch. 
class Dataset(D.Dataset):
	def __init__(self, word2id, train_path):
		self.word2id = word2id
		self.train_path = train_path
		# read the data and label 
		self.data, self.label = reader(train_path)

	def __getitem__(self, index):
		# return the seq and label 
		seq = self.preprocess(self.data[index])
		label = self.label[index]
		return seq, label

	def __len__(self):
		return(len(self.data))

	def preprocess(self, text):
		# used to convert line into tokens and then into their corresponding numericals values using word2id
		line = gensim.utils.simple_preprocess(text)
		seq = []
		for word in line:
			if word in self.word2id:
				seq.append(self.word2id[word])
			else:
				seq.append(self.word2id['<unk>'])
		#convert list into tensor
		seq = torch.from_numpy(np.array(seq))
		return seq

def collate_fn(data):
	'''  

	We should build a custom collate_fn rather than using default collate_fn,
	as the size of every sentence is different and merging sequences (including padding) 
	is not supported in default. 

	Args:
		data: list of tuple (training sequence, label)
	Return:
		padded_seq - Padded Sequence, tensor of shape (batch_size, padded_length)
		length - Original length of each sequence(without padding), tensor of shape(batch_size)
		label - tensor of shape (batch_size)
    '''

    #sorting is important for usage pack padded sequence (used in model). It should be in decreasing order.
	data.sort(key=lambda x: len(x[0]), reverse=True)
	sequences, label = zip(*data)
	length = [len(seq) for seq in sequences]
	padded_seq = torch.zeros(len(sequences), max(length)).long()
	for i, seq in enumerate(sequences):
		end = length[i]
		padded_seq[i,:end] = seq
	return padded_seq, torch.from_numpy(np.array(length)), torch.from_numpy(np.array(label))


#generates the dataloader. 
def dataloader(word2id, train_path, test_path, batch_size = 200):
	train_dataset = Dataset(word2id, train_path)
	test_dataset = Dataset(word2id, test_path)
	train_dataloader = D.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
	test_dataloader = D.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	return train_dataloader, test_dataloader
