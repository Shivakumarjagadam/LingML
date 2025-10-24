from typing import Union

from torch import device, Tensor
from torch.nn import Linear

from .base import BaseModel


class XLNet(BaseModel):

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
        
        outputs = self.base_model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        # https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/models/xlnet/modeling_xlnet.py#L1507C43-L1507C43
        sequence_output = outputs.last_hidden_state
        if self.base_model.sequence_summary.summary_type == 'last':
            pooled_output = sequence_output[:, [-1], :]
        elif self.base_model.sequence_summary.summary_type == 'first':
            pooled_output = sequence_output[:, [0], :]
        outputs.last_hidden_state = pooled_output

        return outputs

    def classifier(
        self,
        output: Tensor
    ):
        
        output = self.base_model.sequence_summary(output)
        logits = self.base_model.logits_proj(output)

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

        return self.base_model.logits_proj

    def set_output(
        self,
        dense: Linear
    ):

        self.base_model.logits_proj = dense