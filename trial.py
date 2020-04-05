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

class RNN(nn.Module):
	def __init__(self,word2id, input_dim, embedding_dim, hidden_dim, output_dim):
		super().__init__()
		self.word2id = word2id
		self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = self.word2id['<pad>'])
		self.rnn = nn.RNN(embedding_dim, hidden_dim)
		self.fc = nn.Linear(hidden_dim, output_dim)
		
	def forward(self, text, length):
		text = text.permute(1,0)
		#text = [sent len, batch size]
		embedded = self.embedding(text)
		#embedded = [sent len, batch size, emb dim]
		print(embedded.shape)
		embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length)

		packed_output, hidden = self.rnn(embedded)
		output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
		print(output.shape)
		#output=[sent len, batch size, hid dim]
		#output over padding token will be zero
		
		#hidden = [1, batch size, hid dim]
		#the last output and hidden should be the same, to check that uncomment below code
		
		# # convert length to index
		# l = [lengths-1 for lengths in length]
		# for i, length in enumerate(l):
		# 	assert torch.equal(output[length,i,:], hidden.squeeze(0)[i])
		
		return self.fc(hidden.squeeze(0))

# train_data, label = reader(train_path)
word2id = build_vocab(train_path)
with open('./word2id.json', 'r') as f:
	word2id = json.load(f)
# print(len(word2id))
train, test = dataloader(word2id, train_path, test_path)
model = RNN(word2id, len(word2id), 100, 256, 1)
# print(len(train))
# print(train)
for pad_seq, length, label in train:
	print(pad_seq.shape, len(pad_seq), len(length), max(length))
	output = model(pad_seq, length)
	exit()