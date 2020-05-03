import numpy as np

STD_SCALER_KEY = 'std_scaler'


def std_scale(df, num_feat, mode, capsule):
    num_x = df[num_feat]

    if mode == 'train':
        scaler = StandardScaler()
        scaler.fit(num_x)

        # register scaler
        capsule[STD_SCALER_KEY] = scaler

    elif mode in ['valid', 'predict']:
        scaler = capsule[STD_SCALER_KEY]

    df[num_feat] = scaler.transform(num_x)

    # transform to np.float32
    df[num_feat] = df[num_feat].astype(np.float32)


class StandardScaler:
    """
    StadardScaler
    - tolerate missing values
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        """
        fit data

        Parameters
        ----------
        x: numpy.nparray
        """
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        return self

    def transform(self, x):
        if self.mean_ is None or self.std_ is None:
            raise ValueError

        scaled_x = (x - self.mean_) / (self.std_ + 1e-6)

        return scaled_x
