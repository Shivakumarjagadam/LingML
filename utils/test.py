from typing import Union

import torch
from torch import Tensor, cat, argmax


def test_batch(model, input_ids: Tensor, attention_mask: Tensor, dense_features: Union[Tensor, None], labels: Tensor):
    
    model.base_model.eval()
    
    with torch.no_grad():
    
        output = model(input_ids=input_ids, attention_mask=attention_mask, dense_features=dense_features, labels=labels)
        loss, logits = torch.sum(output[0]), output[1]
        preds = argmax(logits.detach(), dim=1).cpu()

    return labels.detach().cpu(), preds, loss


def test_epoch(model, data_loader, DEVICE):
    
    epoch_labels, epoch_preds, epoch_loss = Tensor(), Tensor(), 0.
    
    for (input_ids, attention_mask, dense_features, labels) in data_loader:

        input_ids, attention_mask, labels = map(lambda x: x.to(DEVICE, non_blocking=True), (input_ids, attention_mask, labels,))
        if dense_features is not None:
            dense_features = dense_features.to(DEVICE, non_blocking=True)
        labels, preds, loss = test_batch(model, input_ids, attention_mask, dense_features, labels)
        epoch_labels = cat((epoch_labels, labels))
        epoch_preds = cat((epoch_preds, preds))
        epoch_loss += loss

    return epoch_labels, epoch_preds, epoch_loss