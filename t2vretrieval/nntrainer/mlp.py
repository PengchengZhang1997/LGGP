from torch import nn

class MLP(nn.Module):
    '''
    MLP module
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Linear(input_dim, output_dim)
        self.activation_output = nn.GELU()

    def forward(self, x):
        linear_out = self.mlp(x)
        linear_out = self.activation_output(linear_out)

        return linear_out
