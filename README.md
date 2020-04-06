# Sequential-Data-Loader-and-Model-for-Variable-Length-Data
Efficient data loader for text dataset using torch.utils.data.Dataset, collate_fn and torch.utils.data.DataLoader. <br />
Efficient Model for text using torch.nn.utils.rnn.pack_padded_sequence and torch.nn.utils.rnn.pad_packed_sequence. <br />
This Model is used for Sentiment classification on [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/). 
For different dataset you have to modify "reader" function in **data_loader.py** and **vocab.py**.

# Installations Required

* [PyTorch](https://pytorch.org)
* [Gensim](https://radimrehurek.com/gensim/index.html)
* [tqdm](https://github.com/tqdm/tqdm)

# Usage
Put the data in the same folder. 
* To create dictionary -:
```
$ python build_vocab.py
```
* To train the model -:
```
$ python train.py
```
* To just see the dataloader functioning -:
```
Use check_loader.ipynb
```

