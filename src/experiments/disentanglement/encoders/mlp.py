from torch import nn


class MLPEncoder(nn.Module):

    def __init__(
            self, 
            input_dim, 
            hidden_layer_sizes,
            output_dim,
        ) -> None:
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_layer_sizes[0]),
            nn.BatchNorm1d(hidden_layer_sizes[0]),
            nn.ReLU()
        ]
        for i in range(len(hidden_layer_sizes)-1):
            layers.append(nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(hidden_layer_sizes[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_sizes[-1], output_dim))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
        

class MLPFlattenEncoder(MLPEncoder):

    def __init__(self, output_dim, **kwargs) -> None:
        super().__init__(output_dim=output_dim, **kwargs)
        self.size_code = output_dim

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        return self.encoder(x)