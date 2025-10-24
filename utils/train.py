from typing import Union

import torch
from torch import Tensor, cat, argmax


def train_batch(model, optimizer, input_ids: Tensor, attention_mask: Tensor, dense_features: Union[Tensor, None], labels: Tensor):
    
    model.base_model.train()
    optimizer.zero_grad()
    
    output = model(input_ids=input_ids, attention_mask=attention_mask, dense_features=dense_features, labels=labels)
    loss, logits = torch.sum(output[0]), output[1]
    preds = argmax(logits.detach(), dim=1).cpu()

    loss.backward()
    optimizer.step()

    return labels.detach().cpu(), preds, loss


def train_epoch(model, optimizer, data_loader):
    
    epoch_labels, epoch_preds, epoch_loss = Tensor(), Tensor(), 0.
    
    for (input_ids, attention_mask, dense_features, labels) in data_loader:
    
        labels, preds, loss = train_batch(model, optimizer, input_ids, attention_mask, dense_features, labels)
    
        epoch_labels = cat((epoch_labels, labels))
        epoch_preds = cat((epoch_preds, preds))
        epoch_loss += loss

    return epoch_labels, epoch_preds, epoch_loss