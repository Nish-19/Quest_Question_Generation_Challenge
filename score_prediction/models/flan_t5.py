import os
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, T5EncoderModel
from torch.optim import AdamW

from code.score_prediction.batch_collator import CollateWraperScorePredictionFlanT5
from code.score_prediction.models.model_wrapper import ScorePredictionModelWrapper


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.activation = nn.Tanh()
        self.projection = nn.Linear(config.d_model, 1)
        

    def forward(self, hidden_states, attention_mask):
        # Mean pooling of last hidden state ignoring padding tokens: [batch_size X seq_len X hidden_size] -> [batch_size X hidden_size]
        x = torch.div(torch.sum(torch.mul(hidden_states, attention_mask.unsqueeze(-1).float()), dim=1), torch.sum(attention_mask, dim=-1).unsqueeze(-1).float())
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation(x)
        x = self.projection(x)

        return x
    

class ScorePredictionModelFlanT5(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.flan_t5_encoder = T5EncoderModel.from_pretrained(params.lm)
        self.regression_head = RegressionHead(self.flan_t5_encoder.config)
        self.loss_fct = nn.MSELoss()


    def forward(self, **batch):
        outputs = self.flan_t5_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = self.regression_head(outputs.last_hidden_state, batch["attention_mask"])
        loss = self.loss_fct(logits.squeeze(), batch["labels"].squeeze())

        return {
            "loss" : loss, 
            "logits" : logits
            }


class ScorePredictionModelFlanT5Wrapper(ScorePredictionModelWrapper):
    def __init__(self, params, device):
        super().__init__(params, device)
        if( params.model_folder == None ):
            self.tokenizer = AutoTokenizer.from_pretrained(params.lm) 
            self.model = ScorePredictionModelFlanT5(params).to(device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(params.model_folder)
            checkpoint = torch.load(os.path.join(params.model_folder, "model.pt"))
            self.model = ScorePredictionModelFlanT5(params)
            # Overwrite pretrained weights with saved finetuned weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(device)
        # Batch collator function
        self.batch_collator = CollateWraperScorePredictionFlanT5
        

    def set_optimizer(self):
        # TODO P1: change optimizer to AdaFactor
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)


    def set_lr_scheduler(self):
        # LR scheduler
        num_training_steps = len(self.train_loader) * self.params.iters
        num_warmup_steps = self.params.warmup * num_training_steps
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)