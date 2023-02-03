import torch
from torch import nn

from code.utils.load_dataset import load_dataset_score_prediction


class ScorePredictionModelWrapper():
    def __init__(self, params, device):
        self.params = params
        self.device = device


    def prepare_data(self):
        data, self.story_map = load_dataset_score_prediction(self.params.data_folder, self.params.debug)
        self.trainset = data["train"]
        self.valset = data["val"]
        self.testset = data["test"]


    def prepare_dataloaders(self):
        self.train_loader = torch.utils.data.DataLoader(self.trainset, collate_fn=self.batch_collator(self.tokenizer, self.params, self.story_map), 
                                    batch_size=self.params.batch_size, num_workers=self.params.workers, shuffle=True, drop_last=False)                           
        self.val_loader = torch.utils.data.DataLoader(self.valset, collate_fn=self.batch_collator(self.tokenizer, self.params, self.story_map), 
                                    batch_size=self.params.batch_size_eval, num_workers=self.params.workers, shuffle=False, drop_last=False)
        self.test_loader = torch.utils.data.DataLoader(self.testset, collate_fn=self.batch_collator(self.tokenizer, self.params, self.story_map), 
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
        loss = outputs["loss"]
        logits = outputs["logits"]

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
        
        return {
            "loss": loss.detach().cpu(),
            "score_prediction" : {
                "logits": logits.detach().cpu(),
                "labels": batch["labels"].detach().cpu()
                }
            }


    def val_step(self, batch):
        # Same as test step
        out = self.test_step(batch)

        return out


    def test_step(self, batch):
        # TODO P2: we don't need additional code to handle data parallel models?
        with torch.no_grad():
            if( self.params.amp ):
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
            else:
                outputs = self.model(**batch)
        loss = outputs["loss"]
        logits = outputs["logits"]

        if( torch.cuda.device_count() > 1 ):
            loss = loss.mean()

        return {
            "loss": loss.detach().cpu(),
            "score_prediction" : {
                "logits": logits.detach().cpu(),
                "labels": batch["labels"].detach().cpu()
                }
            }


    def set_train_mode(self):
        self.model.train()


    def set_eval_mode(self):
        self.model.eval()