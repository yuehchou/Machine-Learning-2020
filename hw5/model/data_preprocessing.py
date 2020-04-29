import os, cv2, torch, random
import multiprocessing as mp
import numpy as np
from torch.utils.data import DataLoader, Dataset

def load_image(
        path,
        file,
        label
):
    img = cv2.imread(os.path.join(path, file))
    if label:
      return cv2.resize(img,(128, 128)), int(file.split("_")[0])
    return cv2.resize(img,(128, 128))

def readfile(
        path,
        label
):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    # image_dir = sorted(os.listdir(path))
    if not label:
        print('load test')
        image_dir = sorted(os.listdir(path))
    else:
        image_dir = sorted(os.listdir(path))

        temp = image_dir[1]
        image_dir[1:-1] = image_dir[2:]
        image_dir[-1] = temp

    # multiple processing
    pool = mp.Pool()
    multi_res = [pool.apply_async(load_image, (path, file, label)) for file in image_dir]

    if label:
      return np.array([res.get()[0] for res in multi_res]), np.array([res.get()[1] for res in multi_res])
    else:
      return np.array([res.get() for res in multi_res])


def load_image_resnet(
        path,
        file,
        label
):
    img = cv2.imread(os.path.join(path, file))
    if label:
      return cv2.resize(img,(224, 224)), int(file.split("_")[0])
    return cv2.resize(img,(224, 224))

def readfile_resnet(
        path,
        label
):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    # image_dir = sorted(os.listdir(path))
    if not label:
        print('load test')
        image_dir = sorted(os.listdir(path))
    else:
        image_dir = os.listdir(path)
        random.shuffle(image_dir)

    # multiple processing
    pool = mp.Pool()
    multi_res = [pool.apply_async(load_image_resnet, (path, file, label)) for file in image_dir]

    if label:
      return np.array([res.get()[0] for res in multi_res]), np.array([res.get()[1] for res in multi_res])
    else:
      return np.array([res.get() for res in multi_res])

def load_image_inception(
        path,
        file,
        label
):
    img = cv2.imread(os.path.join(path, file))
    if label:
      return cv2.resize(img,(299, 299)), int(file.split("_")[0])
    return cv2.resize(img,(299, 299))

def readfile_inception(
        path,
        label
):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    # image_dir = sorted(os.listdir(path))
    if not label:
        print('load test')
        image_dir = sorted(os.listdir(path))
    else:
        image_dir = os.listdir(path)
        random.shuffle(image_dir)

    # multiple processing
    pool = mp.Pool()
    multi_res = [pool.apply_async(load_image_inception, (path, file, label)) for file in image_dir]

    if label:
      return np.array([res.get()[0] for res in multi_res]), np.array([res.get()[1] for res in multi_res])
    else:
      return np.array([res.get() for res in multi_res])


class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


def cv_split(train_x, train_y, group, n_folds):

    # n_folds >= group >= 1

    group_num = int(len(train_x)/n_folds)

    if group == 1:
        cv_val_x = train_x[:group*group_num]
        cv_val_y = train_y[:group*group_num]
        cv_train_x = train_x[group*group_num:]
        cv_train_y = train_y[group*group_num:]
    elif group == n_folds:
        cv_val_x = train_x[(group - 1)*group_num:]
        cv_val_y = train_y[(group - 1)*group_num:]
        cv_train_x = train_x[:(group - 1)*group_num]
        cv_train_y = train_y[:(group - 1)*group_num]
    else:
        cv_val_x = train_x[(group - 1)*group_num : group*group_num]
        cv_val_y = train_y[(group - 1)*group_num : group*group_num]
        cv_train_x = np.concatenate((
            train_x[:(group - 1)*group_num],
            train_x[group*group_num:]
        ))
        cv_train_y = np.concatenate((
            train_y[:(group - 1)*group_num],
            train_y[group*group_num:]
        ))

    return cv_train_x, cv_train_y, cv_val_x, cv_val_y
