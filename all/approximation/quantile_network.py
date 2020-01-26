import math
import torch
from all import nn
from .approximation import Approximation

class QuantileNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            n_actions,
            n_features,
            embedding_dim=64,
            name='q',
            **kwargs
    ):
        self.n_actions = n_actions
        model = QuantileModule(model, n_features, embedding_dim)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class QuantileModule(nn.Module):
    def __init__(self, model, n_features, embedding_dim=64):
        super().__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        self.device = next(model.parameters()).device
        self.linear = nn.Linear(embedding_dim, n_features).to(next(model.parameters()).device)

    def forward(self, quantiles, states, actions=None):
        # (batch_size x state_features x num_quantiles)
        quantile_features = self._embedding(quantiles)
        # (batch_size x state_features)
        phi = states.features
        # (batch_size x state_features x num_quantiles)
        inputs = states.features[:, :, None] * quantile_features
        # inputs = torch.cat((states.features.unsqueeze(-1).repeat(1, 1, quantile_features.shape[-1]), quantile_features))
        # (batch_size x num_quantiles x state_features)
        inputs = inputs.transpose(1, 2)
        # ((batch_size * num_quantiles) x state_features)
        inputs = inputs.view((-1, inputs.shape[-1]))
        # ((batch_size * num_tau_samples) x num_actions)
        quantile_values = self.model(inputs)
        # (batch_size x num_actions x num_tau_sample
        quantile_values = quantile_values.view((phi.shape[0], -1, quantiles.shape[-1]))
        # terminal quantiles are all 0
        values = quantile_values * states.mask.float().view((-1, 1, 1))
        # return appropriate value
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        return values[torch.arange(len(states)), actions]

    def _embedding(self, quantiles):
        # quantiles = (batch_size x num_quantiles)
        c = math.pi * torch.arange(self.embedding_dim, device=self.device)
        # TODO there has to be a better way
        x = torch.cos(c.expand((quantiles.shape[0], quantiles.shape[1], self.embedding_dim)) * quantiles[:, :, None])
        out = self.linear(x).transpose(1, 2)
        return out
