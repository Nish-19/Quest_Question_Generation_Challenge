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
    for k in outputs[0].keys():
        if( isinstance(outputs[0][k], dict ) ):
            res[k] = {}
            res[k]["labels"] = np.concatenate([tonp(x[k]["labels"]).reshape(-1) for x in outputs])
            res[k]["logits"] = np.concatenate([tonp(x[k]["logits"]).reshape(-1) for x in outputs])
            for metric_name, metric in METRICS:
                res[k][metric_name] = metric(res[k]["labels"], res[k]["logits"])            
        else:
            all_logs = np.concatenate([tonp(x[k]).reshape(-1) for x in outputs])
            res[k] = np.mean(all_logs)
            
    return res