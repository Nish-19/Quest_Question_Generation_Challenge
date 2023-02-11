import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup
from deepspeed.ops.adam import DeepSpeedCPUAdam


class LanguageModel(pl.LightningModule):
    def __init__(self, args, training_dataloader):
        super().__init__()
        self.args = args
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_name)
        # Apply linear probing if required
        if args.linear_probing:
            for name, param in self.model.named_parameters():
                if 'DenseReluDense' in name or 'layer_norm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.model.shared.requires_grad = True
            self.model.lm_head.requires_grad = True
        self.training_dataloader = training_dataloader
    

    def forward(self, input_ids, attention_mask, labels=None):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        return outputs
    

    def common_step(self, batch):
        outputs = self(**batch)
        loss = outputs.loss

        return loss
    

    def training_step(self, batch):
        loss = self.common_step(batch)     
        self.log("training_loss", loss)

        return loss
    

    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.args.learning_rate)
        num_train_optimization_steps = self.args.num_epochs * len(self.training_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
    

    def train_dataloader(self):
        return self.training_dataloader