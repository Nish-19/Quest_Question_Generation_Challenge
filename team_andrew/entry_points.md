## Step 1: Train
```
python -m code.train
```


## Step 2: Prepare model checkpoint
From inside the MODEL_DIR (found in SETTINGS.json), run:
```
python -m code.zero_to_fp32_huggingface \
    --checkpoint_dir model_checkpoint/devout-jazz/last.ckpt/ \
    --output_dir huggingface_model
```


## Step 3: Predict
We would prefer to run:
```
python -m code.predict \
    --num_of_samples 20 \
    --seeds 21
```

However, if there is a GPU out-of-memory issue, please run this instead:
```
python -m code.predict \
    --num_of_samples 5 \
    --seeds 21-0-1-2
```