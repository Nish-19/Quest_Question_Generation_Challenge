"""
Misc helper functions.
"""

import numpy as np


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

    return res