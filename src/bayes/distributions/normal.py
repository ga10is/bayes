import torch
import torch.nn.functional as F


class CustomNormal(torch.distributions.Normal):
    def __init__(self, loc, scale, validate_args=None, eps=1e-6):
        super(CustomNormal, self).__init__(loc, scale, validate_args)
        self.eps = eps

    def __getattribute__(self, name):
        if name == 'scale':
            # return torch.exp(object.__getattribute__(self, name)).clamp(min=self.eps)
            return torch.exp(object.__getattribute__(self, name)) + self.eps
        else:
            return object.__getattribute__(self, name)


class CustomMultivariateNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None):
        super(CustomMultivariateNormal, self).__init__(
            loc, covariance_matrix, precision_matrix, scale_tril, validate_args)
        self.eps = torch.eye(loc.size(0)) * 1e-4

    def __getattribute__(self, name):
        if name == '_unbroadcasted_scale_tril':
            # return F.relu(object.__getattribute__(self, name)) + 1e-4
            # return F.relu(object.__getattribute__(self, name)).clamp(min=1e-8)
            return F.relu(object.__getattribute__(self, name)) + self.eps
            # return torch.exp(object.__getattribute__(self, name))
        else:
            return object.__getattribute__(self, name)
