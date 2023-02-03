import os
import torch
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup

from code.score_prediction.batch_collator import CollateWraperScorePredictionBert
from code.score_prediction.models.model_wrapper import ScorePredictionModelWrapper


class ScorePredictionModelBert(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params    
        # Load pretrained model weights
        self.bert = AutoModelForSequenceClassification.from_pretrained(params.lm, num_labels=1)


    def forward(self, **batch):
        outputs = self.bert(**batch["inputs"], labels=batch["labels"])
        logits = outputs.logits
        loss = outputs.loss

        return {
            "loss" : loss, 
            "logits" : logits
            }


class ScorePredictionModelBertWrapper(ScorePredictionModelWrapper):
    def __init__(self, params, device):
        super().__init__(params, device)
        if( params.model_folder == None ):
            self.tokenizer = AutoTokenizer.from_pretrained(params.lm) 
            self.model = ScorePredictionModelBert(params).to(device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(params.model_folder)
            checkpoint = torch.load(os.path.join(params.model_folder, "model.pt"))
            self.model = ScorePredictionModelBert(params)
            # Overwrite pretrained weights with saved finetuned weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(device)
        # Batch collator function
        self.batch_collator = CollateWraperScorePredictionBert
        
        
    def set_optimizer(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)


    def set_lr_scheduler(self):
        # LR scheduler
        num_training_steps = len(self.train_loader) * self.params.iters
        num_warmup_steps = self.params.warmup * num_training_steps
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)