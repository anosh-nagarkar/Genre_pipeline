import torch
from transformers import PreTrainedModel
from transformers import AutoModel
from .genre_configuration import GenreConfig

class GenreModel(PreTrainedModel):
    config_class = GenreConfig
    def __init__(self,config):
        super().__init__(config)
        self.l1 = AutoModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 17)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1= self.l1(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)['pooler_output']
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
