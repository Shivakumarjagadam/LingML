from typing import Union

from torch import device, Tensor
from torch.nn import Linear

from .base import BaseModel


class BERT(BaseModel):

    def __init__(
        self,
        base_model,
        emb_table_size: Union[int, None],
        dense_size: Union[int, None] = None,
        device: device = device(type='cpu')
    ):

        base_model.pooler = base_model.bert.pooler
        base_model.bert.pooler = None

        super().__init__(base_model, emb_table_size, dense_size, device)

    def transformer(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None
    ):
        
        outputs = self.base_model.bert(input_ids=input_ids, attention_mask=attention_mask)

        return outputs

    def classifier(
        self,
        output: Tensor
    ):
        
        output = self.base_model.pooler(output)
        output = self.base_model.dropout(output)
        logits = self.base_model.classifier(output)

        return logits

    def get_dense(
        self
    ):

        return self.base_model.pooler.dense
    
    def set_dense(
        self,
        dense: Linear
    ):

        self.base_model.pooler.dense = dense

    def get_output(
        self
    ):

        return self.base_model.classifier

    def set_output(
        self,
        dense: Linear
    ):

        self.base_model.classifier = dense