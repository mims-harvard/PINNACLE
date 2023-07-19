import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list, p: float, norm: str, actn: str, order: str = 'nd'):
        super(MLP, self).__init__()
        
        self.n_layer = len(hidden_dims) - 1
        self.in_dim = in_dim
        
        actn2actfunc = {'relu': nn.ReLU(), 'leakyrelu': nn.LeakyReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'selu': nn.SELU(), 'elu': nn.ELU(), 'softplus': nn.Softplus()}
        try:
            actn = actn2actfunc[actn]
        except:
            print(actn)
            raise NotImplementedError

        # Input layer
        layers = [nn.Linear(self.in_dim, hidden_dims[0]), actn]
        
        # Hidden layers
        for i in range(self.n_layer):
            layers += self.compose_layer(in_dim=hidden_dims[i], out_dim=hidden_dims[i+1], norm=norm, actn=actn, p=p, order=order)
        
        # Output layers
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.fc = nn.Sequential(*layers)

    def compose_layer(self, in_dim: int, out_dim: int, norm: str, actn: nn.Module, p: float = 0.0, order: str = 'nd'):
        norm2normlayer = {'bn': nn.BatchNorm1d(in_dim), 'ln': nn.LayerNorm(in_dim), None: None, 'None': None}  # because in_dim is only fixed here
        try:
            norm = norm2normlayer[norm]
        except:
            print(norm)
            raise NotImplementedError
        
        # Options: norm --> dropout or dropout --> norm
        if order == 'nd':
            layers = [norm] if norm is not None else []
            if p != 0:
                layers.append(nn.Dropout(p))
        elif order == 'dn':
            layers = [nn.Dropout(p)] if p != 0 else []
            if norm is not None:
                layers.append(norm)
        else:
            print(order)
            raise NotImplementedError

        layers.append(nn.Linear(in_dim, out_dim))
        if actn is not None:
            layers.append(actn)
        return layers

    def forward(self, x):
        output = self.fc(x)
        return output