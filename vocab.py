import os
import gensim
from collections import Counter
import json

train_path = "./aclImdb/train"
test_path = "./aclImdb/test"

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

def build_vocab(data, min_word_count = 5):
	counter = Counter()
	for line in data:
		l = gensim.utils.simple_preprocess(line)
		counter.update(l)
	#initialise a dictionary or look up table
	word2id = {}
	word2id['<pad>'] = 0
	word2id['<unk>'] = 1
	# include only those in dictionary which have occered more than min word count in the entire data.
	words = [word for word, count in counter.items() if count>min_word_count]

	for i, word in enumerate(words):
		word2id[word] = i+2
	
	with open("word2id.json", 'w') as f:
		json.dump(word2id, f)
	return word2id

data, label = reader(train_path)
word2id = build_vocab(data)
print("Dictionary Formed and saved. The length of dictionary is-: ", len(word2id))
