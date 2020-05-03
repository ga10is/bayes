from sklearn.preprocessing import LabelEncoder

STD_SCALER_KEY = 'std_scaler'
TARGET_ENCODERS_KEY = 'target_encoders'
LABEL_ENCODERS_KEY = 'label_encoders'
EMB_DIMS_KEY = 'emb_dims'


def label_encode(df, categorical_features, mode, capsule):
    if mode == 'train':
        capsule[LABEL_ENCODERS_KEY] = {}
    label_encoders = capsule[LABEL_ENCODERS_KEY]

    for cat_col in categorical_features:

        if mode == 'train':
            label_encoders[cat_col] = LabelEncoder()
            label_encoders[cat_col].fit(df[cat_col])

        df[cat_col] = label_encoders[cat_col].transform(df[cat_col])


def get_emb_dims(df, categorical_features, mode, capsule):
    if mode == 'train':
        cat_dims = [int(df[col].nunique()) for col in categorical_features]
        emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

        capsule[EMB_DIMS_KEY] = emb_dims
    elif mode in ['valid', 'predict']:
        pass


def smooth_target_encode(df, ste_feat, mode, capsule):
    if mode == 'train':
        capsule[TARGET_ENCODERS_KEY] = {}
    tes = capsule[TARGET_ENCODERS_KEY]

    for feat in ste_feat:
        if mode == 'train':
            tes[feat] = TargetEncoder('y', feat)
            tes[feat].fit(df)

        mean_col = '%s_mean' % feat
        n_col = '%s_n' % feat
        df[mean_col], df[n_col] = tes[feat].transform(df[feat])


class TargetEncoder:
    def __init__(self, target_col, cat_col):
        self.target_col = target_col
        self.cat_col = cat_col

    def fit(self, df):
        gb = df.groupby(self.cat_col)
        # mean of target for each category, the number of data for each category
        self.mean_dict_ = gb.mean()[self.target_col].to_dict()
        self.size_dict_ = gb.size().to_dict()

        # mean of target for all data
        self.total_mean_ = df[self.target_col].mean()
        # mean and std of the number of data for each category
        self.n_mean_ = gb.size().mean()
        self.n_std_ = gb.size().std()

        return self

    def transform(self, series):
        mean_x = series.map(self.mean_dict_)
        n_x = series.map(self.size_dict_)
        return mean_x, n_x
