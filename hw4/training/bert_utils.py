import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# max_length_sentence = tokenizer.max_len_single_sentence
max_length_sentence = 50

def load_training_data(path='training_label.txt'):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip('\n') for line in lines]

        x = [tokenizer.tokenize(line[10:]) for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip('\n') for line in lines]

        x = [tokenizer.tokenize(line) for line in lines]
        return x

def load_testing_data(path='testing_data'):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]

    X = [tokenizer.tokenize(line) for line in lines]
    return X

def convert_training_data(
        load_fptah,
        save_fpath,
        label=True
):
    data={}
    if label:
        with open(load_fptah, 'r') as f:
            lines = f.readlines()
        data['label'] = [line[0] for line in lines]
        data['text'] = [line.strip('\n')[10:] for line in lines]
        df = pd.DataFrame(data)
        df.to_csv(save_fpath, index=False)
    else:
        with open(load_fptah, 'r') as f:
            lines = f.readlines()
        data['text'] = [line.strip('\n') for line in lines]
        df = pd.DataFrame(data)
        df.to_csv(save_fpath, index=True)

def convert_testing_data(
        load_fptah,
        save_fpath,
):
    data={}
    with open(load_fptah, 'r') as f:
        lines = f.readlines()
    data['text'] = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
    df = pd.DataFrame(data)
    df.to_csv(save_fpath, index=True)

def convert_text_to_numpy(
        text_list,
        max_length=max_length_sentence,
        save_np_text_fpath=None,
        label=None,
        save_np_label_fpath=None
):
    row = len(text_list)
    np_data = np.zeros((row, max_length))

    for i in range(row):
        temp = tokenizer.encode_plus(
            text_list[i],
            max_length=max_length,
            token_type_ids=False,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            return_token_type_ids=False,
            pad_to_max_length=True
        )
        np_data[i] = temp['input_ids']

    if save_np_text_fpath:
        np.save(save_np_text_fpath, np_data)
        print('Save numpy array to {}'.format(save_np_text_fpath))

    if label:
        if save_np_label_fpath:
            np.save(save_np_label_fpath, np.array(label))
            print('Save label to {}'.format(save_np_label_fpath))
        return np_data, np.array(label)
    return np_data

class TextDataset(Dataset):
    def __init__(self, x, y=None):
        # self.x = x
        self.x = torch.LongTensor(x)
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            # self.y = torch.LongTensor(y)
            self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
