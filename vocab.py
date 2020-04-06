import os
import gensim
from collections import Counter
import json

train_path = "./aclImdb/train"
test_path = "./aclImdb/test"

def build_vocab(path, min_word_count = 5):
	pos_path = os.path.join(path, "pos")
	neg_path = os.path.join(path, "neg")
	#two folders, positive review and negative review
	data = []
	label = []
	counter = Counter()
	for file in os.listdir(pos_path):
		f = open(os.path.join(pos_path, file))
		line = f.read()
		# we tokenize and preprocess the line using gensim
		line = gensim.utils.simple_preprocess(line) 
		data.append(line)
		#used for building dictionary
		counter.update(line)

	for file in os.listdir(neg_path):
		f = open(os.path.join(neg_path, file))
		line = f.read()
		line = gensim.utils.simple_preprocess(line) 
		data.append(line)
		counter.update(line)

	#initialise a dictionary or look up table
	word2id = {}
	word2id['<pad>'] = 0
	word2id['<unk>'] = 1
	# print(data[:1])
	# include only those in dictionary which have occered more than min word count in the entire data.
	words = [word for word, count in counter.items() if count>min_word_count]

	for i, word in enumerate(words):
		word2id[word] = i+2
	
	with open("word2id.json", 'w') as f:
		json.dump(word2id, f)
	return word2id

word2id = build_vocab(train_path)