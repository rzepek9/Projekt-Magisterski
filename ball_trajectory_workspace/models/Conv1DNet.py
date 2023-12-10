import torch


class Conv1DNet(torch.nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        hidden_size: int,
        num_classes: int,
        dropout: float,
        activation_function: str = 'relu',
        n_conv_layers: int = 3,
        conv_neurons: list = [16, 32, 64],
        kernel_sizes: list = [3, 5, 3],
        skip_connection: bool = False,
    ):
        super(Conv1DNet, self).__init__()

        assert (
            n_conv_layers == len(conv_neurons) == len(kernel_sizes)
        ), print(f"The length of conv_neurons and kernel_sizes list must be equal to n_conv_layers,\
                 but got {len(conv_neurons)}, {len(kernel_sizes)}, {n_conv_layers}")

        self.hidden_size = hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.skip_connection = skip_connection

        if activation_function == "relu":
            self.activation_function = torch.nn.ReLU()
        elif activation_function == "leakyrelu":
            self.activation_function = torch.nn.LeakyReLU()
        elif activation_function == "elu":
            self.activation_function = torch.nn.ELU()
        elif activation_function == "prelu":
            self.activation_function = torch.nn.PReLU()
        elif activation_function == "gelu":
            self.activation_function = torch.nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

        self.convs = torch.nn.ModuleList()
        for i in range(n_conv_layers):
            in_channels = input_shape[0] if i == 0 else conv_neurons[i - 1]
            out_channels = conv_neurons[i]
            kernel_size = kernel_sizes[i]
            self.convs.append(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
            )
        self.batch_norms = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=out_channels) for out_channels in conv_neurons
        ])

        sample_input = torch.zeros(1, *input_shape)
        sample_output = self._forward_convs(sample_input, apply_batch_norm=False)
        output_shape = torch.prod(torch.tensor(sample_output.shape)).item()

        if skip_connection is True:
            input_size = output_shape + torch.prod(torch.tensor(input_shape)).item()

        else:
            input_size = output_shape

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(input_size, self.hidden_size),
            self.activation_function,
            self.dropout,
            torch.nn.Linear(self.hidden_size, num_classes),
        )

    def _forward_convs(self, x, apply_batch_norm: bool = True) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if apply_batch_norm:
                x = self.batch_norms[i](x)
            x = self.activation_function(x)
            x = self.dropout(x)
        return x

    def forward(self, x) -> torch.Tensor:
        orig_x = x.view(x.size(0), -1)

        x = self._forward_convs(x)

        x = x.view(x.size(0), -1)

        if self.skip_connection:
            x = torch.cat(
                (orig_x, x), dim=1
            )

        return self.MLP(x)
