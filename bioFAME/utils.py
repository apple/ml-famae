import os
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from tqdm import tqdm


def set_random_seed(seed):
    """Use with caution under multi-gpu setting"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_acc(network, loader, device, test_aug=None):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.long().to(device)

            if test_aug:
                x = test_aug(x)

            logits = network(x)
            correct += logits.argmax(1).eq(y).float().sum().item()
            total += y.shape[0]

    network.train()
    return correct / total


def collect_cm(network, loader, device, test_aug=None):
    
    pred_collect = []
    target_collect = []

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.long().to(device)

            if test_aug:
                x = test_aug(x)

            logits = network(x)
            pred = logits.argmax(dim=1, keepdim=True)
            target_pred = y

            pred_collect.extend(torch.flatten(pred.detach().cpu()))
            target_collect.extend(torch.flatten(target_pred.detach().cpu()))
    
    network.train()

    pred_c = np.array(pred_collect)
    target_c = np.array(target_collect)

    precision = precision_score(target_c, pred_c, average='macro')
    recall = recall_score(target_c, pred_c, average='macro')
    F1 = f1_score(target_c, pred_c, average='macro')

    conf_matrix = confusion_matrix(target_c, pred_c)
    bal_binary = 0
    for index in range(conf_matrix.shape[0]):
        tp = conf_matrix[index, index]
        tn = float(np.sum(conf_matrix[:index, :index]) + np.sum(conf_matrix[index + 1 :, index + 1 :]))
        tn += float(np.sum(conf_matrix[:index, index + 1 :])) + float(np.sum(conf_matrix[index + 1 :, :index]))
        fp = float(np.sum(conf_matrix[:index, index]) + np.sum(conf_matrix[index + 1 :, index]))
        fn = float(np.sum(conf_matrix[index, :index]) + np.sum(conf_matrix[index, index + 1 :]))

        bal_binary += (tn + tp) / (tp + tn + fp + fn)
    bal_binary = bal_binary/conf_matrix.shape[0]

    return precision, recall, F1, conf_matrix, bal_binary


class linear_clf_set(Dataset):
    '''takes latent representation for SSL MLP training'''
    def __init__(self, data):

        self.reps = data[0]
        self.labels = data[1].long()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.reps[idx, :], self.labels[idx]


class linear_clf_train(object):
    def __init__(self, net, classifier, optimizer, train_dataloader, test_dataloader, device = "cpu", batch_size=1024,
                 num_epochs = 10, disable_tqdm = False, writer=None, writer_tag = "", val_dataloader=None):
        
        self.net = net
        self.net.eval()

        self.classifier = classifier
        self.optimizer = optimizer
        self.writer = writer
        self.tag = writer_tag

        self.disable_tqdm = disable_tqdm
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.data_train = linear_clf_set(self.compute_representations(train_dataloader))
        self.data_test = linear_clf_set(self.compute_representations(test_dataloader))

        self.best_number = 0
        self.train_linear_layer()

        self.train_acc = self.compute_accuracy(DataLoader(self.data_train, batch_size=batch_size))
        self.test_acc = self.compute_accuracy(DataLoader(self.data_test, batch_size=batch_size))

    def compute_representations(self, dataloader):
        """ store the representations
        :param net: ResNet or smth
        :param dataloader: train_loader and test_loader
        """
        #self.net.eval()
        reps, labels = [], []

        for i, (x, label) in enumerate(dataloader):
            # load data
            x = x.to(self.device)
            labels.append(label)

            # forward
            with torch.no_grad():
                representation = self.net(x, latent_mode=True)
                reps.append(representation.detach().cpu())

            if i % 100 == 0:
                reps = [torch.cat(reps, dim=0)]
                labels = [torch.cat(labels, dim=0)]

        reps = torch.cat(reps, dim=0)
        labels = torch.cat(labels, dim=0)
        #self.net.train()
        return (reps, labels)

    def compute_accuracy(self, dataloader):
        self.classifier.eval()
        right = []
        total = []
        for x, label in dataloader:
            x, label = x.to(self.device), label.to(self.device)
            # feed to network and classifier
            with torch.no_grad():
                pred_logits = self.classifier(x)
            # compute accuracy
            _, pred_class = torch.max(pred_logits, 1)
            right.append((pred_class == label).sum().item())
            total.append(label.size(0))
        self.classifier.train()
        return sum(right) / sum(total)

    def train_linear_layer(self):
        progress_bar = tqdm(range(self.num_epochs), disable=self.disable_tqdm, position=0, leave=True)
        for epoch in progress_bar:
            for x, label in DataLoader(self.data_train, batch_size=self.batch_size):
                self.classifier.train()
                
                x, label = x.to(self.device), label.to(self.device)
                pred_class = self.classifier(x)
                loss = F.cross_entropy(pred_class, label)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            curr_number = self.compute_accuracy(DataLoader(self.data_test, batch_size=self.batch_size))
            if curr_number >= self.best_number:
                self.best_number = curr_number

            if self.writer is not None:
                self.writer.log_metrics({'CLFtraining/val-tag{}'.format(self.tag): curr_number}, step = epoch)

            progress_bar.set_description('Linear_CLF Epoch: [{}/{}] Acc@1:{:.3f}% BestAcc@1:{:.3f}%'
                                         .format(epoch, self.num_epochs, curr_number, self.best_number))
