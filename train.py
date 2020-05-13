from data_loader import dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from tqdm import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class RNN(nn.Module):
	def __init__(self,word2id, input_dim, embedding_dim, hidden_dim, output_dim):
		super().__init__()
		self.word2id = word2id
		#input dimension is lenght of your dictionary
		self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = self.word2id['<pad>'])
		self.rnn = nn.RNN(embedding_dim, hidden_dim)
		self.fc = nn.Linear(hidden_dim, output_dim)
		
	def forward(self, text, length):
		#text = [batch size, sent len]

		text = text.permute(1,0)
		#text = [sent len, batch size]
		embedded = self.embedding(text)
		#embedded = [sent len, batch size, emb dim]
		
		# since we have output of different length with zero padded, when we use pack padded sequence then LSTM or RNN will only process non paded elements of our sequence.
		# The RNN will return a packed output (which is nothing but hidden state at all non paded elements) as well as the last hidden state of our element.  
		# Without packed padded sequences, hidden is tensors from the last element in the sequence, which will most probably be a pad token,
		# however when using packed padded sequences they are both from the last non-padded element in the sequence.
		embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, length)

		packed_output, hidden = self.rnn(embedded)
		output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
		#output=[sent len, batch size, hid dim]
		#output over padding token will be zero
		
		#hidden = [1, batch size, hid dim]
		#the last output and hidden should be the same, to check that uncomment below code
		
		# # convert length to index
		# l = [lengths-1 for lengths in length]
		# for i, length in enumerate(l):
		# 	assert torch.equal(output[length,i,:], hidden.squeeze(0)[i])
		out = self.fc(hidden.squeeze(0))
		# No softmax as we are using BCEWithLogitsLoss
		return out

def accuracy(prediction, labels):
	rounded_preds = torch.round(torch.sigmoid(prediction))
	correct = (rounded_preds == labels).float() #convert into float for division 
	acc = correct.sum() / len(correct)
	return acc


def train(train_data, model, optimizer, criterion):
	avg_loss = 0
	avg_acc = 0
	# print(next(model.parameters()).is_cuda)
	model.train()
	for pad_seq, length, label in tqdm(train_data):
		
		optimizer.zero_grad()
		
		pad_seq = pad_seq.to(device)
		label = label.to(device)
		length = length.to(device)
		label = label.type(torch.cuda.FloatTensor)
		
		# print(label, pad_seq.is_cuda, label.is_cuda)
		output = model(pad_seq, length)
		#output =[batch_size, 1]
		output = output.reshape(output.size(0))
		loss = criterion(output, label)
		loss.backward()
		optimizer.step()

		acc = accuracy(output, label)
		avg_loss += loss.item()
		avg_acc += acc.item()
		
	return (1.0 * avg_loss)/len(train_data), (1.0*avg_acc)/len(train_data) 

def evaluate(test_data, model, criterion):
	avg_loss = 0
	avg_acc = 0
	# print(next(model.parameters()).is_cuda)
	model.eval()
	for pad_seq, length, label in tqdm(test_data):
		pad_seq = pad_seq.to(device)
		label = label.to(device)
		length = length.to(device)
		label = label.type(torch.cuda.FloatTensor)		
		# print(label, pad_seq.is_cuda, label.is_cuda)
		output = model(pad_seq, length)
		#output =[batch_size, 1]
		output = output.reshape(output.size(0))
		loss = criterion(output, label)

		acc = accuracy(output, label)
		avg_loss += loss.item()
		avg_acc += acc.item()
		
	return (1.0 * avg_loss)/len(test_data), (1.0*avg_acc)/len(test_data) 

if __name__ == '__main__':

	train_path = "./aclImdb/train"
	test_path = "./aclImdb/test"

	with open('./word2id.json', 'r') as f:
		word2id = json.load(f)
	
	train_data, test_data = dataloader(word2id, train_path, test_path)
	model = RNN(word2id, len(word2id), 100, 256, 1)
	optimizer = optim.SGD(model.parameters(), lr=1e-3)
	criterion = nn.BCEWithLogitsLoss()
	

	num_epochs = 20
	for i in range(num_epochs):
		print("Training")
		model.to(device)
		criterion.to(device)
		train_loss, train_acc = train(train_data, model, optimizer, criterion)
		print("Evaluating")
		eval_loss, eval_acc = evaluate(test_data, model, criterion)
		print("Training loss: {}, Evaluation loss: {}, Training accuracy: {}, Evlaution accuracy: {}".
			format(train_loss, eval_loss, train_acc, eval_acc))