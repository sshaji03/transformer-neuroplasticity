import torch
import torch.nn as nn
from dataclasses import dataclass
from transformers import BertModel

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
HIDDEN_SIZE = 128


@dataclass
class ModelOutput:
    """Holds model results with loss and logits before softmax"""
    loss: torch.Tensor
    logits: torch.Tensor


class BertWithTaskHeads(nn.Module):
    """Multi-head BERT model, underlying BERT weights are the same but final layer classification
     head is used so that the right number of classes is outputted for the current task."""
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        # stores all the task heads so can activate easily
        self.heads = nn.ModuleDict()
        self.active_task = None

    def add_task_head(self, task_name, num_classes):
        """populates task names and heads for appropriate number of classes"""
        safe_name = task_name.replace("/", "_")
        self.heads[safe_name] = nn.Linear(HIDDEN_SIZE, num_classes)

    def set_task(self, task_name):
        """Sets a new task to active"""
        self.active_task = task_name.replace("/", "_")

    def forward(self, input_ids, attention_mask, labels=None):
        """bert encodes the input and active head gets fed information encoded in CLS token to calculate loss and logits"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]
        logits = self.heads[self.active_task](cls)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ModelOutput(loss=loss, logits=logits)
