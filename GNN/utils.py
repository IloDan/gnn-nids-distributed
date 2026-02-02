def get_qfs_target_feature_k(df, percentages=None):
    if percentages is None:
        percentages = [0.25, 0.5, 0.75]
    n_features = df["dataset_features"].iloc[0]
    return [int(n_features * percentage) for percentage in percentages]


def get_qfs_target_feature_k_rows(df, percentages=None):
    if percentages is None:
        percentages = [0.25, 0.5, 0.75]
    return [
        df[df["target_feature_k"] == target_feature_k]
        for target_feature_k in get_qfs_target_feature_k(df, percentages)
    ]
