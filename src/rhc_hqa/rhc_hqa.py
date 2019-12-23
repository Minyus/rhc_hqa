import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMRegressor


def cv_index_list_to_df(d):
    out_dict = {}
    for group, index_list in d.items():
        for index in index_list:
            out_dict[index] = group
    df = pd.DataFrame.from_dict(out_dict, orient="index", columns=["split"])
    df.index.name = "id"
    return df


def estimate_cate(df, parameters):
    col_outcome = parameters.get("col_outcome")
    col_treatment = parameters.get("col_treatment")
    cols_feature = parameters.get("cols_feature")

    propensity_lower = parameters.get("propensity_lower")
    propensity_upper = parameters.get("propensity_upper")

    split_list = df["split"].drop_duplicates().to_list()
    split_list = sorted(split_list)

    ps_model_dict = {}
    tot_model_dict = {}

    np.random.seed(42)

    for s in split_list:
        ps_model = LogisticRegression(solver="liblinear", random_state=42)
        tot_model = LGBMRegressor(
            min_child_samples=400, importance_type="gain", random_state=42
        )
        col_propensity = "propensity_score_{}".format(s)
        col_trans_outcome = "transformed_outcome_{}".format(s)
        col_cate = "cate_{}".format(s)

        train_df = df.query("split != @s")

        ps_model.fit(train_df[cols_feature], train_df[col_treatment])
        df[col_propensity] = ps_model.predict_proba(df[cols_feature])[:, 1]

        df[col_propensity].clip(
            inplace=True, lower=propensity_lower, upper=propensity_upper
        )

        df[col_trans_outcome] = (
            df[col_outcome]
            * (df[col_treatment] - df[col_propensity])
            / (df[col_propensity] * (1 - df[col_propensity]))
        )

        train_df = df.query("split != @s")

        tot_model.fit(train_df[cols_feature], train_df[col_trans_outcome])
        df[col_cate] = tot_model.predict(df[cols_feature])

        col_cate_if_seps_1 = "cate_if_seps_1_{}".format(s)
        col_cate_if_seps_0 = "cate_if_seps_0_{}".format(s)
        col_cate_diff_seps = "cate_diff_seps_{}".format(s)

        seps_1_df = df[cols_feature].copy()
        seps_1_df["seps_1"] = 1.0
        df[col_cate_if_seps_1] = tot_model.predict(seps_1_df)

        seps_0_df = df[cols_feature].copy()
        seps_0_df["seps_1"] = 0.0
        df[col_cate_if_seps_0] = tot_model.predict(seps_0_df)

        df[col_cate_diff_seps] = df[col_cate_if_seps_1] - df[col_cate_if_seps_0]

        imp_df = pd.DataFrame(
            {
                "feature": cols_feature,
                "propensity_model_coef": np.squeeze(ps_model.coef_),
                "cate_model_importances": tot_model.feature_importances_,
            }
        )
        ps_model_dict[s] = ps_model
        tot_model_dict[s] = tot_model
    model_dict = dict(ps=ps_model_dict, tot=tot_model_dict)

    return df, imp_df, model_dict
