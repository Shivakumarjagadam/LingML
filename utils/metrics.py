from collections import defaultdict

import torch
from torch import Tensor, where


class Results:

    def __init__(self):

        self.results = defaultdict(lambda: defaultdict(list))

    def update(self, split: str, batch: int, labels: Tensor, preds: Tensor, loss: Tensor):

        pos_pred_labels, neg_pred_labels = labels[where(preds == 1)[0]], labels[where(preds == 0)[0]]
        tp, fn = torch.sum(pos_pred_labels).item(), torch.sum(neg_pred_labels).item()

        self.results[split]['batch'].append(batch)
        self.results[split]['loss'].append(loss.item())
        self.results[split]['tp'].append(tp)
        self.results[split]['fp'].append(pos_pred_labels.shape[0]-tp)
        self.results[split]['fn'].append(fn)
        self.results[split]['tn'].append(neg_pred_labels.shape[0]-fn)

    def metrics(self, split: str, last: int):

        tp = sum(self.results[split]['tp'][-last:])
        fp = sum(self.results[split]['fp'][-last:])
        fn = sum(self.results[split]['fn'][-last:])
        tn = sum(self.results[split]['tn'][-last:])
        loss = sum(self.results[split]['loss'][-last:])

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1_score = (tp + tp) / (tp + tp + fp + fn)

        return accuracy, f1_score, loss