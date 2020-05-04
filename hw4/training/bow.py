import torch
import numpy as np
# bow.py
class BOW():
    def __init__(self, max_len=10000):
        self.wordfreq = {}
        self.vector_size = max_len
        self.word2idx = {}
    def bow(self, train_sentences, test_sentences):
        for sentence in train_sentences + test_sentences:
            for word in sentence:
                if word in self.wordfreq.keys(): self.wordfreq[word] += 1
                else: self.wordfreq[word] = 1
        self.wordfreq = sorted(self.wordfreq.items(), key=lambda x: x[1], reverse=True)
        if self.vector_size > len(self.wordfreq): self.vector_size = len(self.wordfreq)
        for idx, (word, freq) in enumerate(self.wordfreq):
            if idx == self.vector_size: break
            self.word2idx[word] = len(self.word2idx)
        self.train_bow_list = np.zeros((len(train_sentences), self.vector_size))
        self.test_bow_list = np.zeros((len(test_sentences), self.vector_size))
        for idx, sentence in enumerate(train_sentences):
            for word in sentence:
                if word in self.word2idx.keys():
                    self.train_bow_list[idx][self.word2idx[word]] += 1
        for idx, sentence in enumerate(test_sentences):
            for word in sentence:
                if word in self.word2idx.keys():
                    self.test_bow_list[idx][self.word2idx[word]] += 1
    def __getitem__(self, data_type):
        if data_type == 'train':
            return torch.FloatTensor(self.train_bow_list)
        elif data_type == 'test':
            return torch.FloatTensor(self.test_bow_list)
