from collections import OrderedDict

import torch
from torch import nn


class InferenceNetwork(nn.Module):
    """Inference Network that supports combined and zeroshot architectures."""

    def __init__(
        self,
        input_size,
        bert_size=None,
        output_size=None,
        hidden_sizes=None,
        activation=nn.Softplus(),
        dropout=0.2,
        inference_type="combined",
        norm=False,
    ):
        """
        Initialize InferenceNetwork.

        Args:
            input_size : int, dimension of input.
            bert_size : int, dimension of BERT embeddings.
            output_size : int, dimension of output.
            hidden_sizes : tuple, length = n_layers.
            activation : string, 'softplus' or 'relu', default 'softplus'.
            dropout : float, default 0.2.
            inference_type : string, 'combined' or 'zeroshot', default 'combined'.
        """
        super(InferenceNetwork, self).__init__()

        self.input_size = input_size
        self.bert_size = bert_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.inference_type = inference_type
        self.activation = activation
        self.norm = norm

        if self.inference_type == "combined":
            self.input_layer = nn.Linear(
                input_size + input_size, hidden_sizes[0])
            self.adapt_bert = nn.Linear(bert_size, input_size)
        elif self.inference_type == "zeroshot":
            self.adapt_bert = nn.Linear(bert_size, hidden_sizes[0])
        elif self.inference_type == "avitm":
            self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        self.hiddens = nn.Sequential(
            OrderedDict(
                [
                    (
                        "l_{}".format(i),
                        nn.Sequential(nn.Linear(h_in, h_out), self.activation),
                    )
                    for i, (h_in, h_out) in enumerate(
                        zip(hidden_sizes[:-1], hidden_sizes[1:])
                    )
                ]
            )
        )

        self.f_mu = nn.Linear(hidden_sizes[-1], output_size)
        self.f_mu_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.f_sigma = nn.Linear(hidden_sizes[-1], output_size)
        self.f_sigma_batchnorm = nn.BatchNorm1d(output_size, affine=False)

        self.dropout_enc = nn.Dropout(p=self.dropout)

    def forward(self, x):
        """Forward pass."""
        if self.inference_type == "combined":
            x_bert = self.adapt_bert(x["embedding"])
            x = torch.cat((x["bow"], x_bert), 1)
            x = self.input_layer(x)
        elif self.inference_type == "zeroshot":
            x_bert = self.adapt_bert(x["embedding"])
            x = x_bert
        elif self.inference_type == "avitm":
            if self.norm:
                x = {"bow": x["bow"] / (x["bow"].sum(1, keepdim=True) + 1e-12)}
            x = self.input_layer(x["bow"])

        x = self.activation(x)
        x = self.hiddens(x)
        x = self.dropout_enc(x)
        mu = self.f_mu_batchnorm(self.f_mu(x))
        log_sigma = self.f_sigma_batchnorm(self.f_sigma(x))

        return mu, log_sigma
