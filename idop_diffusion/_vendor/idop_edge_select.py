# -*- coding: utf-8 -*-
"""Unmodified edge_select excerpt from the existing local idop_core.py.

Original author: Yu Wang (yuwang@bimsa.cn).
Upstream: https://github.com/Yu-Wang-0923/idopNetworkAPP
MIT license: LICENSE.idopnetwork alongside this file.
Only candidate-edge selection is vendored; no curve fitting or ODE weights.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

def edge_select(
    data_ecg: pd.DataFrame,
    alpha: float = 0.1, # LASSO regularization parameter
    k: int = 10,
    threshold: float = 0.9,
) -> pd.DataFrame:
    """Select edges from fitted idopNetwork data.

    Note
    ----
    从数据中恢复稀疏有向图.
    对每个 target 变量, 其支撑集来源于除 target 变量以外的所有变量.
    每个时间窗内对 x(逐列) 与 y 做 z-score 标准化后再拟合 LASSO,
    系数还原回原始尺度后判断是否为 0.

    Parameters
    ----------
    data_ecg : pd.DataFrame
        Shape = (n_timepoints, n_channels)
        columns = ["time", "feature1", "feature2", ..., "featureN"]
    alpha : float, optional
        LASSO regularization parameter. The default is 0.5.
    k : int, optional
        Number of roughly equal windows to split the idopNetwork into. The default is 10.
    threshold : float, optional
        超过 90 % 以上的 window 中都出现的 source 变量 才会进入支撑集.
        

    Returns
    -------
    edge_supports : pd.DataFrame
        Columns = ["target", "source"]
        Example:
            target | source
            I      | {II,aVR}
            II     | {aVR,aVF}

    """
    lead_columns = [column for column in data_ecg.columns if column != "time"]
    if len(lead_columns) < 2:
        raise ValueError("data_ecg must contain at least two lead columns")
    if alpha <= 0:
        raise ValueError("alpha must be greater than 0")
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > len(data_ecg):
        raise ValueError("k cannot exceed the number of time points")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    windows = np.array_split(data_ecg[lead_columns].to_numpy(dtype=float), k, axis=0)
    edge_counts = {
        target: {source: 0 for source in lead_columns if source != target}
        for target in lead_columns
    }

    for window in windows:
        for target_idx, target in enumerate(lead_columns):
            source_indices = [i for i in range(len(lead_columns)) if i != target_idx]
            x = window[:, source_indices]
            y = window[:, target_idx]

            # 标准化 x(逐列) 与 y，让 LASSO 目标尺度 O(1)，改善收敛
            x_mean = x.mean(axis=0)
            x_scale = x.std(axis=0)
            x_scale[x_scale == 0] = 1.0
            x_std = (x - x_mean) / x_scale

            y_mean = float(y.mean())
            y_scale = float(y.std())
            if y_scale == 0.0:
                y_scale = 1.0
            y_std = (y - y_mean) / y_scale

            model = Lasso(alpha=alpha, fit_intercept=True, max_iter=100_000, tol=1e-5)
            model.fit(x_std, y_std)

            # 系数还原到原始尺度: beta_orig = beta_std * y_scale / x_scale
            coefficients = model.coef_ * (y_scale / x_scale)
            for local_idx, coefficient in enumerate(coefficients):
                if coefficient != 0.0:
                    source = lead_columns[source_indices[local_idx]]
                    edge_counts[target][source] += 1

    rows = []
    for target in lead_columns:
        selected = [
            source
            for source in lead_columns
            if source != target and edge_counts[target][source] / k > threshold
        ]
        rows.append({"target": target, "source": "{" + ",".join(selected) + "}"})

    return pd.DataFrame(rows, columns=["target", "source"])
