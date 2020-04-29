from collections import OrderedDict
from functools import reduce

import torch
import torch.nn as nn
import torch.distributions as torchdist
from bayes.distributions.flow import FlowDistribution


class VIModel(nn.Module):
    def __init__(self, model):
        super(VIModel, self).__init__()
        self.loc_dict, n_dim = self.extract_param(model)

        self.mu = nn.Parameter(torch.zeros(n_dim))
        self.log_sigma = nn.Parameter(torch.zeros(n_dim))

        self.dist = FlowDistribution(loc=self.mu, scale=self.log_sigma)

    def extract_param(self, model):
        start = 0
        loc_dict = OrderedDict()
        for n, p in model.named_parameters():
            num = reduce(lambda x, y: x * y, p.size())
            end = start + num
            loc_info = {
                'size': p.size(),
                'loc': (start, end)
            }
            start = end

            loc_dict[n] = loc_info

        n_dim = start
        print(loc_dict)
        print(n_dim)
        return loc_dict, n_dim


def normal_prior_dist(model):
    n_dim = 0
    for n, p in model.named_parameters():
        num = reduce(lambda x, y: x * y, p.size())
        n_dim += num

    mu = torch.zeros(n_dim)
    # log_sigma = torch.zeros(n_dim)
    # dist = CustomNormal(loc=mu, scale=log_sigma)
    sigma = torch.ones(n_dim) * 10
    dist = torchdist.Normal(loc=mu, scale=sigma)
    # sigma = torch.eye(n_dim) * 10
    # dist = torchdist.MultivariateNormal(loc=mu, scale_tril=sigma)
    # dist = CustomMultivariateNormal(loc=mu, scale_tril=sigma)
    return dist


def transform_param(w, loc_dict):
    param_dict = OrderedDict()
    for n, loc_info in loc_dict.items():
        loc = loc_info['loc']
        size = loc_info['size']
        sample = w[slice(*loc)]
        param_dict[n] = sample.view(*size)

    return param_dict
