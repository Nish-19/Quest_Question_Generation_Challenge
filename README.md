To create a local train-val split from the public train file run this command from `code`:
```
python -m code.utils.create_dataset_split
```

The hash of your data splits will be compared against the original stored in `utils/data_hash.json`