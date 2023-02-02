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
        self.tokenizer = AutoTokenizer.from_pretrained(params.lm) 
        self.config = AutoConfig.from_pretrained(params.lm, num_labels=1)
        self.model = AutoModelForSequenceClassification.from_pretrained(params.lm, config=self.config)


    def forward(self, **batch):
        outputs = self.model(**batch["inputs"], labels=batch["labels"])
        logits = outputs.logits
        loss = outputs.loss

        return {
            "loss" : loss, 
            "logits" : logits
            }


class ScorePredictionModelBertWrapper(ScorePredictionModelWrapper):
    def __init__(self, params, device):
        super().__init__(params, device)
        self.model = ScorePredictionModelBert(params).to(device)
        # Batch collator function
        self.batch_collator = CollateWraperScorePredictionBert
        self.prepare_data()
        self.dataloaders()
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)
        # LR scheduler
        num_training_steps = len(self.train_loader) * params.iters
        num_warmup_steps = params.warmup * num_training_steps
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)