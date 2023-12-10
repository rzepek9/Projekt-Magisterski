import torch
from torch.nn.utils.rnn import pad_packed_sequence


class LSTM(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        layer_size: int,
        n_classes: int,
        dropout: float,
        bidirectional: bool,
        activation_function: str = "relu",
    ):
        super(LSTM, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.n_classes = n_classes
        self.bidirectional = bidirectional

        self.dropout = torch.nn.Dropout(dropout)

        self.lstm_1 = torch.nn.LSTM(
            self.n_features,
            self.hidden_size,
            self.layer_size,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        lstm_2_input_size = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.lstm_2 = torch.nn.LSTM(
            lstm_2_input_size,
            self.hidden_size,
            self.layer_size,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

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

        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2 if self.bidirectional else hidden_size, self.hidden_size),
            torch.nn.Dropout(dropout),
            self.activation_function,
            torch.nn.Linear(self.hidden_size, self.n_classes),
        )

    def forward(self, packed_inputs):
        num_directions = 2 if self.bidirectional else 1

        hidden_state = torch.zeros(
            self.layer_size * num_directions,
            packed_inputs.batch_sizes[0],
            self.hidden_size,
        ).to(packed_inputs.data.device)

        cell_state = torch.zeros(
            self.layer_size * num_directions,
            packed_inputs.batch_sizes[0],
            self.hidden_size,
        ).to(packed_inputs.data.device)

        packed_output, _ = self.lstm_1(packed_inputs, (hidden_state, cell_state))
        packed_output = self.dropout(packed_output.data)
        packed_output = torch.nn.utils.rnn.PackedSequence(packed_output, packed_inputs.batch_sizes)
        packed_output, _ = self.lstm_2(packed_output)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = output[:, -1, :]
        return self.MLP(output)
