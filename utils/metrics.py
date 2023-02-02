from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr, kendalltau


def rmse_metric(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def mae_metric(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def pcc_metric(y_true, y_pred):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    pcc, _ = pearsonr(y_true, y_pred)

    return pcc


def scc_metric(y_true, y_pred):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    scc, _ = spearmanr(y_true, y_pred)

    return scc


def kcc_metric(y_true, y_pred):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    kcc, _ = kendalltau(y_true, y_pred)

    return kcc


def r2_metric(y_true, y_pred):
    return r2_score(y_true, y_pred)