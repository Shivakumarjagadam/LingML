from typing import Union

from torch import device, Tensor
from torch.nn import Linear

from .base import BaseModel


class XLM(BaseModel):

    def __init__(
        self,
        base_model,
        emb_table_size: Union[int, None],
        dense_size: Union[int, None] = None,
        device: device = device(type='cpu')
    ):

        base_model.config.initializer_range = base_model.config.init_std

        super().__init__(base_model, emb_table_size, dense_size, device)

    def transformer(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None
    ):

        outputs = self.base_model.transformer(input_ids=input_ids, attention_mask=attention_mask)

        return outputs

    def classifier(
        self,
        output: Tensor
    ):
        
        logits = self.base_model.sequence_summary(output)

        return logits

    def get_dense(
        self
    ):

        return self.base_model.sequence_summary.summary
    
    def set_dense(
        self,
        dense: Linear
    ):

        self.base_model.sequence_summary.summary = dense

    def get_output(
        self
    ):

        return self.base_model.sequence_summary.summary

    def set_output(
        self,
        dense: Linear
    ):

        self.base_model.sequence_summary.summary = dense