### from https://github.com/JieyuZ2/wrench/blob/af9b77fb919abd57a9bb6f9e49a5febb61cf6a9a/wrench/backbone.py#L122
import torch.nn as nn
from wrench.backbone import BERTBackBone

class BertTextClassifier(BERTBackBone):
    """
    Bert with a MLP on top for text classification
    """

    def __init__(self, n_class, model_name='bert-base-cased', fine_tune_layers=-1, max_tokens=512, binary_mode=False, **kwargs):
        super(BertTextClassifier, self).__init__(n_class=n_class, model_name=model_name, fine_tune_layers=fine_tune_layers, binary_mode=binary_mode)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)
        self.max_tokens = max_tokens
        self.hidden_size = self.config.hidden_size

    def forward(self, batch, return_features=False):  # inputs: [batch, t]
        device = self.get_device()
        outputs = self.model(input_ids=batch["input_ids"].to(device), attention_mask=batch['mask'].to(device))
        h = self.dropout(outputs.pooler_output)
        output = self.classifier(h)
        if return_features:
            return output, h
        else:
            return output