'''
A few caveats:
- this only works with deepspeed stage 1 trained/fine-tuned model
- this only works with T5ForConditionalGeneration model for now
'''

import os, argparse, torch
from transformers import T5ForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument('-CP', '--ckpt_path', help='deepspeed checkpoint path')
parser.add_argument('-SP', '--save_path', help='save path for converted deepspeed ckpt')
ckpt_path = parser.parse_args().ckpt_path
save_path = parser.parse_args().save_path

# Load the huggingface model
model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')

# Load the deepspeed checkpoint
state_dict = torch.load(ckpt_path)

# Map the deepspeed checkpoint state_dict keys into huggingface compatible format
new_state_dict = {}
for k in state_dict['module'].keys():
    v = state_dict['module'][k]
    new_state_dict[k.replace('_forward_module.model.', '')] = v

# Load the state_dict into huggingface model
model.load_state_dict(new_state_dict)

# IMPORTANT STEP: Save the model
# From my experience, this step has to be done to ensure proper GPU memory usage
# i.e., in the inference code, you would directly load the model from the save_path, using
# model = T5ForConditionalGeneration.from_pretrained(save_path)
# Otherwise, if you use the model from `model.load_state_dict(new_state_dict)`,
# where new_state_dict is the state_dict loaded from the deepspeed checkpoint,
# it uses twice as much as GPU memory as the model loaded from the save_path
model.save_pretrained(save_path)