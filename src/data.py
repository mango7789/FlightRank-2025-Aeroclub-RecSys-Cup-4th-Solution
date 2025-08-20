import os
import numpy as np
import polars as pl
import xgboost as xgb
import lightgbm as lgb


def split_dataset(
    train: pl.DataFrame,
    X: pl.DataFrame,
    y: pl.DataFrame,
    groups: pl.DataFrame,
    cat_features_final,
    model,
):
    X = X.with_columns(
        [
            (pl.col(c).rank("dense") - 1).fill_null(-1).cast(pl.Int16)
            for c in cat_features_final
        ]
    )

    # Create train/validation/test dataset
    n1 = 16487352  # split train to train and val (10%) in time
    n2 = train.height
    X_tr, X_va, X_te, X_fl = X[:n1], X[n1:n2], X[n2:], X[:n2]
    y_tr, y_va, y_te, y_fl = y[:n1], y[n1:n2], y[n2:], y[:n2]
    groups_tr, groups_va, groups_te, groups_fl = (
        groups[:n1],
        groups[n1:n2],
        groups[n2:],
        groups[:n2],
    )

    if model == "xgboost":

        # weight = pl.read_parquet("./data/weight.parquet")

        # weight_tr = (
        #     weight.join(
        #         train.slice(0, n1).select(["Id", "ranker_id"]),
        #         on=["Id", "ranker_id"],
        #         how="inner",
        #     )
        #     .select("group_weight")
        #     .to_numpy()
        # )

        # weight_va = (
        #     weight.join(
        #         train.slice(n1, n2 - n1).select(["Id", "ranker_id"]),
        #         on=["Id", "ranker_id"],
        #         how="inner",
        #     )
        #     .select("group_weight")
        #     .to_numpy()
        # )

        # weight_fl = (
        #     weight.join(
        #         train.select(["Id", "ranker_id"]), on=["Id", "ranker_id"], how="inner"
        #     )
        #     .select("group_weight")
        #     .to_numpy()
        # )

        group_sizes_tr = (
            groups_tr.group_by("ranker_id", maintain_order=True)
            .agg(pl.len())["len"]
            .to_numpy()
        )
        group_sizes_va = (
            groups_va.group_by("ranker_id", maintain_order=True)
            .agg(pl.len())["len"]
            .to_numpy()
        )
        group_sizes_te = (
            groups_te.group_by("ranker_id", maintain_order=True)
            .agg(pl.len())["len"]
            .to_numpy()
        )
        group_sizes_fl = (
            groups_fl.group_by("ranker_id", maintain_order=True)
            .agg(pl.len())["len"]
            .to_numpy()
        )

        dtrain = xgb.DMatrix(
            X_tr,
            label=y_tr,
            group=group_sizes_tr,
            # weight=weight_tr,
            feature_names=X.columns,
        )
        dval = xgb.DMatrix(
            X_va,
            label=y_va,
            group=group_sizes_va,
            # weight=weight_va,
            feature_names=X.columns,
        )
        dtest = xgb.DMatrix(
            X_te, label=y_te, group=group_sizes_te, feature_names=X.columns
        )
        dfull = xgb.DMatrix(
            X_fl,
            label=y_fl,
            group=group_sizes_fl,
            # weight=weight_fl,
            feature_names=X.columns,
        )
    elif model == "lightgbm":

        def get_group_sizes(groups):
            return (
                groups.group_by("ranker_id", maintain_order=True)
                .agg(pl.len())["len"]
                .to_list()
            )

        group_sizes_tr = get_group_sizes(groups_tr)
        group_sizes_va = get_group_sizes(groups_va)
        group_sizes_te = get_group_sizes(groups_te)
        group_sizes_fl = get_group_sizes(groups_fl)

        # Convert to pandas dataframe for compability
        X_tr = X_tr.to_pandas()
        X_va = X_va.to_pandas()
        X_te = X_te.to_pandas()
        X_fl = X_fl.to_pandas()

        dtrain = lgb.Dataset(
            X_tr,
            label=y_tr.to_numpy().flatten(),
            group=group_sizes_tr,
            feature_name=X.columns,
            categorical_feature=cat_features_final,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_va,
            label=y_va.to_numpy().flatten(),
            group=group_sizes_va,
            feature_name=X.columns,
            categorical_feature=cat_features_final,
            free_raw_data=False,
        )
        dtest = lgb.Dataset(
            X_te,
            label=y_te.to_numpy().flatten(),
            group=group_sizes_te,
            feature_name=X.columns,
            categorical_feature=cat_features_final,
            free_raw_data=False,
        )
        dfull = lgb.Dataset(
            X_fl,
            label=y_fl.to_numpy().flatten(),
            group=group_sizes_fl,
            feature_name=X.columns,
            categorical_feature=cat_features_final,
            free_raw_data=False,
        )
    else:
        raise ValueError("Unsupported Models!")

    return (
        dtrain,
        dval,
        dtest,
        dfull,
        X_tr,
        y_tr,
        groups_tr,
        X_va,
        y_va,
        groups_va,
        X_te,
    )
