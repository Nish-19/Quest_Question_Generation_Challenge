"""
Misc helper functions.
"""

import numpy as np

from code.utils.metrics import rmse_metric, mae_metric, pcc_metric, scc_metric, kcc_metric, r2_metric


METRICS = [("rmse", rmse_metric), ("mae", mae_metric), ("pcc", pcc_metric), ("scc", scc_metric), ("kcc", kcc_metric), ("r2", r2_metric)]


def tonp(x):
    if isinstance(x, (np.ndarray, float, int)):
        return np.array(x)
    return x.detach().cpu().numpy()


def agg_all_metrics(outputs):    
    res = {}
    keys = [k for k in outputs[0].keys() if not isinstance(outputs[0][k], dict)]
    for k in keys:
        all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
        if k != 'epoch':
            res[k] = np.mean(all_logs)
        else:
            res[k] = all_logs[-1]

    if( "score_prediction" in outputs[0] ):
        label_logs = np.concatenate([tonp(x["labels"]).reshape(-1) for x in outputs])
        pred_logs = np.concatenate([tonp(x["logits"]).reshape(-1) for x in outputs])
        for metric_name, metric in METRICS:
            res[metric_name] = metric(label_logs, pred_logs)

    return res