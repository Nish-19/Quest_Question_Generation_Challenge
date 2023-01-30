import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup, T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
import os

from code.t5.batch_collator import CollateWraperGenerative
from code.utils.load_dataset import load_dataset


class QuestionGenerationModel(nn.Module):
    def __init__(self, params, saved_models_dir):
        super().__init__()
        self.params = params
        if( params.checkpoint ):
            model_folder = os.path.join(saved_models_dir, params.model_folder)
            self.tokenizer = T5Tokenizer.from_pretrained(model_folder)
            self.model = T5ForConditionalGeneration.from_pretrained(model_folder, return_dict=True)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(params.lm) 
            self.model = T5ForConditionalGeneration.from_pretrained(params.lm) 


    def forward(self, **batch):
        outputs = self.model(**batch)
        
        return outputs


class QuestionGenerationModelWrapper():
    def __init__(self, params, device, saved_models_dir):
        self.params = params
        self.device = device
        self.prepare_data()
        self.model = QuestionGenerationModel(params, saved_models_dir).to(device)
        self.dataloaders()
        # TODO: change optimizer to AdaFactor
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)
        # LR scheduler
        num_training_steps = len(self.train_loader) * params.iters
        num_warmup_steps = params.warmup * num_training_steps
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)


    def prepare_data(self):
        data, self.story_map = load_dataset(self.params.data_folder, self.params.cross_val_fold, self.params.debug, self.params.augment, self.params.data_augment_folder)
        self.trainset = data["train"]
        self.valset = data["val"]


    def dataloaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.trainset, collate_fn=CollateWraperGenerative(self.model.tokenizer, self.params, self.story_map), 
                                    batch_size=self.params.batch_size, num_workers=self.params.workers, shuffle=True, drop_last=False)                           
        self.val_loader = torch.utils.data.DataLoader(self.valset, collate_fn=CollateWraperGenerative(self.model.tokenizer, self.params, self.story_map), 
                                    batch_size=self.params.batch_size_eval, num_workers=self.params.workers, shuffle=False, drop_last=False)


    def zero_grad(self):
        self.optimizer.zero_grad()


    def grad_step(self, scaler):
        if( self.params.amp ):
            scaler.unscale_(self.optimizer)
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)   
            scaler.step(self.optimizer)
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  
            self.optimizer.step()
        self.lr_scheduler.step()


    def train_step(self, batch, scaler):
        self.zero_grad()
        if( self.params.amp ):
            # Casts operations to mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
        else:
            outputs = self.model(**batch)
        loss = outputs.loss

        # Multi-GPU training
        if( torch.cuda.device_count() > 1 ):
            # TODO P2: is this the correct way for multi-GPU? resolve multi gpu warning
            loss = loss.mean()

        # Scales the loss and calls backward() to create scaled gradients
        if( self.params.amp ):
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.grad_step(scaler)
        # Updates the scale for next iteration
        if( self.params.amp ):
            scaler.update()

        return {"loss": loss.detach().cpu()}


    def val_step(self, batch):
        # TODO P2: we don't need additional code to handle data parallel models?
        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
        loss = outputs.loss

        if( torch.cuda.device_count() > 1 ):
            loss = loss.mean()

        return {"loss": loss.detach().cpu()}


    def set_train_mode(self):
        self.model.train()
        self.model.model.train()


    def set_eval_mode(self):
        self.model.eval()
        self.model.model.eval()