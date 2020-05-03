from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import LabelEncoder

STD_SCALER_KEY = 'std_scaler'
TARGET_ENCODERS_KEY = 'target_encoders'
LABEL_ENCODERS_KEY = 'label_encoders'
EMB_DIMS_KEY = 'emb_dims'


def label_encode(df, categorical_features, mode, capsule):
    if mode == 'train':
        capsule[LABEL_ENCODERS_KEY] = OrderedDict()
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


def smooth_target_encode(df, target_col, ste_cols, mode, capsule):
    if mode == 'train':
        capsule[TARGET_ENCODERS_KEY] = OrderedDict()
    target_encoders = capsule[TARGET_ENCODERS_KEY]

    for col in ste_cols:
        if mode == 'train':
            encoder = TargetEncoder(target_col, col)
            encoder.fit(df)
            target_encoders[col] = encoder
        else:
            encoder = target_encoders[col]

        mean_col = '%s_mean' % col
        n_col = '%s_n' % col
        df[mean_col], df[n_col] = encoder.transform(df[col])


class TargetEncoder:
    def __init__(self, target_col, cat_col):
        self.target_col = target_col
        self.cat_col = cat_col

    def fit(self, df):
        # mean of target for all data
        self.total_mean_ = df[self.target_col].mean()

        gb = df.groupby(self.cat_col)
        # mean and std of the number of data for each category
        n_mean = gb.size().mean()
        n_std = gb.size().std()
        if np.isnan(n_std):
            raise ValueError('std of the number of categorical data is nan.')
        normalized_size = (gb.size() - n_mean) / n_std

        # mean of target for each category, the number of data for each category
        self.mean_dict_ = gb.mean()[self.target_col].to_dict()
        self.normalized_size_dict_ = normalized_size.to_dict()

        return self

    def transform(self, series):
        mean_x = series.map(self.mean_dict_)
        normalized_size_x = series.map(self.normalized_size_dict_)
        return mean_x, normalized_size_x
