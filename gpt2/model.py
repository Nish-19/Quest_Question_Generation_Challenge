"""
GPT-2 model.
"""

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch import nn
from torch.optim import AdamW
import copy


from code.utils.load_dataset import load_dataset
from code.gpt2.batch_collator import CollateWraperGenerative


class LanguageModelBase(nn.Module):
    def __init__(self, params, device):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(params.lm, return_dict=True)


    def forward(self, **kwargs):
        return self.model(**kwargs)


class LanguageModelBaseWrapper():
    def __init__(self, params, device):
        super().__init__()
        self.params = copy.deepcopy(params)
        self.device = device
        self.model = LanguageModelBase(self.params, self.device).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.params.lr)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.params.lm)
        # Add pad token to be the same as end of text token
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def prepare_data(self):
        data, story_map = load_dataset(self.params.data_folder, self.params.debug)
        self.trainset = data['train']
        self.valset = data['val']
        self.story_map = story_map


    def dataloaders(self):
        CollateWraper = CollateWraperGenerative
        train_loader = torch.utils.data.DataLoader(self.trainset, collate_fn=CollateWraper(self.tokenizer, self.params, self.story_map, self.params.lm_loss_location), 
                                    batch_size=self.params.batch_size, num_workers=self.params.workers, shuffle=True, drop_last=True)
        # For validation set, compute CLM loss only on question tokens                            
        val_loader = torch.utils.data.DataLoader(self.valset, collate_fn=CollateWraper(self.tokenizer, self.params, self.story_map, lm_loss_location = "question"), 
                                    batch_size=self.params.batch_size, num_workers=self.params.workers, shuffle=False, drop_last=True)
        
        return train_loader, val_loader


    def zero_grad(self):
        self.optimizer.zero_grad()


    def grad_step(self, scaler):
        if( self.params.amp ):
            scaler.step(self.optimizer)
        else:
            self.optimizer.step()


    def train_step(self, batch, scaler):
        self.zero_grad()
    
        if( self.params.amp ):
            # Casts operations to mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch["inputs"])
        else:
            outputs = self.model(**batch["inputs"])
        loss = outputs.loss

        # Multi gpu training
        if( torch.cuda.device_count() > 1 ):
            # TODO P2: is this the correct way? resolve warning multi gpu warning
            # scales the loss, and calls backward() to create scaled gradients
            if( self.params.amp ):
                scaler.scale(loss.mean()).backward()
            else:
                loss.mean().backward()
        else:
            # Scales the loss, and calls backward() to create scaled gradients
            if( self.params.amp ):
                scaler.scale(loss).backward()
            else:
                loss.backward()

        self.grad_step(scaler)
        # updates the scale for next iteration
        if( self.params.amp ):
            scaler.update()
        
        if( torch.cuda.device_count() > 1 ):
            loss = loss.mean()

        return {
            'loss': loss.detach().cpu()
            }


    def eval_step(self, batch):
        # Same as in test step in LanguageModelBase
        out = self.test_step(batch)

        return out


    def test_step(self, batch):
        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch["inputs"])
            else:
                outputs = self.model(**batch["inputs"])
        loss = outputs.loss

        if( torch.cuda.device_count() > 1 ):
            loss = loss.mean()

        return {
            'loss': loss.detach().cpu()
            }


    def set_train_mode(self):
        self.model.train()


    def set_eval_mode(self):
        self.model.eval()