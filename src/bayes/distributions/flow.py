import torch
import torch.nn as nn

from bayes.distributions.normal import CustomNormal


"""
Module for Normalizing Flows
"""


class PlanarFlow(nn.Module):
    def __init__(self, n, u_normalize=False):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, n).normal_(0, 0.01))
        self.w = nn.Parameter(torch.randn(1, n).normal_(0, 0.01))
        self.b = nn.Parameter(torch.zeros(1))

        torch.nn.init.kaiming_normal_(self.u)
        torch.nn.init.kaiming_normal_(self.w)

        self.u_normalize = u_normalize

    def forward(self, z):
        u_hat = self.norm_u() if self.u_normalize else self.u

        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        f_z = f_z.squeeze()
        return f_z

    def log_det(self, z):
        u_hat = self.norm_u() if self.u_normalize else self.u

        psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) * self.w
        det = 1 + u_hat @ psi.t()
        log_abs_det_jacobians = torch.log(torch.abs(det) + 1e-6).squeeze()

        return log_abs_det_jacobians

    def norm_u(self):
        wtu = (self.w @ self.u.t()).squeeze()
        m_wtu = - 1 + torch.log1p(wtu.exp())
        u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        return u_hat


class FlowDistribution(nn.Module):
    def __init__(self, loc, scale):
        super().__init__()

        self.normal = CustomNormal(loc=loc, scale=scale)
        n_dims = loc.size(0)
        self.flows = nn.ModuleList(
            [PlanarFlow(n_dims, u_normalize=False) for _ in range(32)])

    def sample(self):
        with torch.no_grad():
            zs = self.rsample()
        return zs

    def rsample(self):
        zs = []
        z = self.normal.rsample()

        zs.append(z)
        for flow in self.flows:
            z = flow(z)
            zs.append(z)

        return zs

    def log_prob(self, zs):
        # zs
        z_0 = zs[0]
        z_ks = zs[1:]

        if len(self.flows) != len(z_ks):
            raise ValueError

        sum_log_det = 0
        for flow, z in zip(self.flows, z_ks):
            sum_log_det += flow.log_det(z)

        return self.normal.log_prob(z_0).sum() - sum_log_det
