RANDOM_STATE = 42


def get_hyper_params(model):
    best_params = None

    if model == "xgboost":
        # best_params = {
        #     "learning_rate": 0.023688639503640783,
        #     "max_depth": 12,
        #     "min_child_weight": 17,
        #     "subsample": 0.8394633936788146,
        #     "colsample_bytree": 0.4936111842654619,
        #     "gamma": 0.7799726016810132,
        #     "lambda": 1.2551115172973832,
        #     "alpha": 2.9154431891537547,
        #     "objective": "rank:pairwise",
        #     "eval_metric": "ndcg@3",
        #     "seed": RANDOM_STATE,
        #     "n_jobs": 32,
        #     "device": "cuda",
        #     "tree_method": "hist",
        # }
        # best_params = {
        #     "learning_rate": 0.022641389657079056,
        #     "max_depth": 14,
        #     "min_child_weight": 2,
        #     "subsample": 0.8842234913702768,
        #     "colsample_bytree": 0.45840689146263086,
        #     "gamma": 3.3084297630544888,
        #     "lambda": 6.952586917313028,
        #     "alpha": 0.6395254133055179,
        #     "objective": "rank:pairwise",
        #     "eval_metric": "ndcg@3",
        #     "seed": RANDOM_STATE,
        #     "n_jobs": 32,
        #     "device": "cuda",
        # }
        best_params = {
            "learning_rate": 0.03168911713420175,
            "max_depth": 14,
            "min_child_weight": 26,
            "subsample": 0.9998786532443618,
            "colsample_bytree": 0.3057471543313584,
            "gamma": 0.5155201382200343,
            "lambda": 52.096256305667175,
            "alpha": 0.09601451692342958,
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@3",
            "seed": RANDOM_STATE,
            "n_jobs": 32,
            "device": "cuda",
        }
    elif model == "lightgbm":
        # best_params = {
        #     "learning_rate": 0.04,
        #     "num_leaves": 256,
        #     "max_depth": 15,
        #     "feature_fraction": 0.5,
        #     "bagging_fraction": 0.8,
        #     "bagging_freq": 5,
        #     "lambda_l1": 0.1,
        #     "lambda_l2": 0.5,
        #     "num_boost_round": 600,
        #     "objective": "lambdarank",
        #     "metric": "ndcg",
        #     "ndcg_eval_at": [3],
        #     "seed": RANDOM_STATE,
        #     # "device": "gpu",
        #     "verbosity": -1,
        #     "n_jobs": 32,
        # }
        best_params = {
            "boosting_type": "gbdt",
            "learning_rate": 0.029951044628883604,
            "num_leaves": 362,
            "max_depth": 20,
            "feature_fraction": 0.37386560505124056,
            "bagging_fraction": 0.8804853582153065,
            "bagging_freq": 4,
            "min_child_samples": 78,
            "min_split_gain": 0.12307359245482069,
            "lambda_l1": 0.5398456444519547,
            "lambda_l2": 1.3427796821854638,
            "max_bin": 339,
            "extra_trees": False,
            "num_boost_round": 750,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [3],
            "seed": RANDOM_STATE,
            # "device": "gpu",
            "verbosity": 1,
            "n_jobs": 32,
        }

    return best_params
