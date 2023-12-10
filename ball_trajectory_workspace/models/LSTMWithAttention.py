import torch
from torch.nn.utils.rnn import pad_packed_sequence


class Attention(torch.nn.Module):
    def __init__(self, hidden_size: int, bidirectional: bool) -> None:
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.attn = torch.nn.Linear(
            self.hidden_size * 4 if self.bidirectional else self.hidden_size * 2,
            self.hidden_size,
        )
        self.v = torch.nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> torch.Tensor:
        attn_energies = torch.tanh(
            self.attn(
                torch.cat(
                    (hidden.repeat(1, encoder_outputs.size(1), 1), encoder_outputs),
                    dim=2,
                )
            )
        )
        attn_energies = attn_energies.transpose(1, 2)
        v = self.v.repeat(attn_energies.size(0), 1).unsqueeze(1)
        attn_energies = torch.bmm(v, attn_energies).squeeze(1)
        return torch.nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)


class LSTMWithAttention(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        layer_size: int,
        n_classes: int,
        dropout: float,
        bidirectional: bool = False,
        activation_function: str = "relu",
    ) -> None:
        super(LSTMWithAttention, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.layer_size = layer_size
        self.n_classes = n_classes
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = torch.nn.LSTM(
            self.n_features,
            self.hidden_size,
            self.layer_size,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        self.attention = Attention(hidden_size, self.bidirectional)

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
            self.activation_function,
            torch.nn.Dropout(dropout),
            torch.nn.Linear(self.hidden_size, self.n_classes),
        )

    def forward(self, packed_inputs: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:
        packed_output, _ = self.lstm(packed_inputs)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        attention_weights = self.attention(
            output[
                :,
                -1,
                : self.hidden_size * 2 if self.bidirectional else self.hidden_size,
            ].unsqueeze(1),
            output,
        )
        context_vector = torch.bmm(attention_weights, output).squeeze(1)

        return self.MLP(context_vector)
