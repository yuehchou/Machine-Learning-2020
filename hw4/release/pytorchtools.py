import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        # self.val_loss_min = np.Inf
        self.val_acc_max = 0
        self.delta = delta

    # def __call__(self, val_loss, model, path):
    def __call__(self, val_acc, model, path):

        # score = -val_loss
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, path)
            self.save_checkpoint(val_acc, model, path)
        # elif score < self.best_score + self.delta:
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model, path)
            self.save_checkpoint(val_acc, model, path)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model, path):
    def save_checkpoint(self, val_acc, model, path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
            print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_acc_max, val_acc))
        torch.save(model.state_dict(), path)
        # self.val_loss_min = val_loss
        self.val_acc_max = val_acc
