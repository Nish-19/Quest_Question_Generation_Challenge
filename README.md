To create a local train-val split from the public train file run this command from `code`:
```
python -m code.utils.create_dataset_split
```

The hash of your data splits will be compared against the original stored in `utils/data_hash.json`

### New: added deepspeed integration for model paralellism
TL:DR: this allows you to train billion-scale parameter model 
on multiple GPUs by distributing the model parameters
on multiple GPUs.

__Prerequisites__
Install deepspeed

### Usecase: see command below (basically nothing changed)

`CUDA_VISIBLE_DEVICES=0,2 python t5_finetune.py -N flan-t5-xxl -M google/flan-t5-xxl -D 2`

