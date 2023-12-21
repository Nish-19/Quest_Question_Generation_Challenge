NOTE: This repository is outdated and no longer maintained. For updates please check out - https://github.com/umass-ml4ed/question-gen-aug-ranking

## Finetuning

To finetune an encoder-decoder model (T5/BART):

```
python -m code.finetune.finetune_org \
    -W -MT T -MN google/flan-t5-large \
    -N flan_t5_large
```

The code accepts a list of arguments which are defined in the ```add_params``` function. 

The trained model checkpoint gets saved in the ```Checkpoints_org``` folder. 

## Inference/Generation

To get the inference/ generation using a pre-trained model: 

```
python -m code.finetune.inference_org \
    -MT T -MN google/flan-t5-large \
    -N flan_t5_large -DS N \
    -PS 0.9 -NS 10
```

The csv file containing the generations are saved in ```results_org```

## Finetuning Distribution Ranking-Based Model 
```
python -m code.ranking_kl.bert_rank \
    -W -Attr -ExIm -MN YituTech/conv-bert-base \
    -SN convbert_org_10_0.001_0.01 \
    -alpha1 0.001 -alpha2 0.01
```

## Predictions from Distribution Ranking-Based Model
```
python -m code.ranking_kl.bert_rank_inference \
    -Attr -ExIm -MN YituTech/conv-bert-base \
    -SN convbert_org_10_0.001_0.01
```

The csv file containing the generations are saved in ```results_rank_kl```


## ROUGE Scores

To compute the ROUGE Scores

```
python -m code.utils.compute_rouge_score \
    --eval_folder results_org \
    --eval_filename flan_t5_large_org.csv
```

