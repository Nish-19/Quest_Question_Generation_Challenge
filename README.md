## Splitting Data

To create a local train-val split from the public train file run this command from `code`:
```
python -m code.utils.create_dataset_split
```

The hash of your data splits will be compared against the original stored in `utils/data_hash.json`

## Finetuning

To finetune an encoder-decoder model (T5/BART):

```
python -m code.finetune.finetune \
    -MT T \
    -MN t5-small \
    -N t5_small
```

The code accepts a list of arguments which are defined in the ```add_params``` function. 

The trained model checkpoint gets saved in the ```Checkpoints``` folder. 

## Inference/Generation

To get the inference/ generation using a pre-trained model: 

```
python -m code.finetune.inference -N t5_small -M t5-small
```

The csv file containing the generations are saved in ```results/train_val_split_csv```

## BLEURT Scores

To compute the BLEURT Scores

```
python -m code.utils.compute_eval_metric \
    --eval_folder train_val_split_csv \
    --eval_filename text-curie-001_20221103-030617.csv \
    --batch_size 64
```

