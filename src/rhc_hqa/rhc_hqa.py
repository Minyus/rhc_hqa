import pandas as pd
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

    for s in split_list:
        ps_model = LogisticRegression(solver="liblinear")
        tot_model = LGBMRegressor(min_child_samples=400)
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
        cate_sr = tot_model.predict(df[cols_feature])
        df[col_cate] = cate_sr
    return df
