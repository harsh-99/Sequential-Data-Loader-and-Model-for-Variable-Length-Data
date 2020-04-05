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

