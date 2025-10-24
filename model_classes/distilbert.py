from typing import Union

from torch import device, Tensor
from torch.nn import Linear, ReLU

from .base import BaseModel


class DistilBERT(BaseModel):

    def __init__(
        self,
        base_model,
        emb_table_size: Union[int, None],
        dense_size: Union[int, None] = None,
        device: device = device(type='cpu')
    ):

        super().__init__(base_model, emb_table_size, dense_size, device)

    def transformer(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None
    ):

        return self.base_model.distilbert(input_ids=input_ids, attention_mask=attention_mask)

    def classifier(
        self,
        output: Tensor
    ):
        
        output = self.base_model.pre_classifier(output[:, 0, :])
        output = ReLU()(output)
        output = self.base_model.dropout(output)
        logits = self.base_model.classifier(output)

        return logits

    def get_dense(
        self
    ):

        return self.base_model.pre_classifier
    
    def set_dense(
        self,
        dense: Linear
    ):

        self.base_model.pre_classifier = dense

    def get_output(
        self
    ):

        return self.base_model.classifier

    def set_output(
        self,
        dense: Linear
    ):

        self.base_model.classifier = dense