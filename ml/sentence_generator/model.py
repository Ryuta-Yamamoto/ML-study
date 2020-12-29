# %%
import torch
from torch import random
from torch.nn import (
    Module,
    TransformerEncoderLayer,
    TransformerEncoder,
    Linear,
    Embedding,
    ReLU,
    LSTM,
    Sequential,
    Parameter,
    ModuleList,
)
from torch.nn.utils import spectral_norm
from torch.tensor import Tensor


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAEModule(Module):
    def generate(self, n_sentenses):
        raise NotImplementedError


# %%
def set_lstm_spectral_norm(lstm: LSTM) -> LSTM:
    for n in range(lstm.num_layers):
        spectral_norm(lstm, f'weight_ih_l{n}')
        spectral_norm(lstm, f'weight_hh_l{n}')


def normal_random_like(tensor):
        return torch.normal(torch.zeros_like(tensor), torch.ones_like(tensor))
    

class TransformerModule(Module):
    def __init__(
            self, 
            n_dim: int,
            n_layers: int = 3,
            n_head: int = 8,
    ):
        super().__init__()
        self.enc = TransformerEncoder(
            TransformerEncoderLayer(n_dim, n_head),
            num_layers=n_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 0)
        return self.enc(x_t).transpose(1, 0)


# %%
lstm = LSTM(input_size=1, hidden_size=2, num_layers=3)

# %%
class LSTMDecoder(Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
    ) -> None:
        super().__init__()
        self.layer = LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        set_lstm_spectral_norm(self.layer)

    def forward(self, initial_input, hidden_cell_tuple, length):
        tensors = []
        previous = initial_input
        for _ in range(length):
            output, hidden_cell_tuple = self.layer(previous, hidden_cell_tuple)
            tensors.append(output)
            previous = output
        return torch.cat(tensors, axis=1)


class VAE(VAEModule):
    def __init__(self, hidden_size, num_gaussian):
        super().__init__()
        self.num_gaussian = num_gaussian
        self.mu_input = Linear(hidden_size, num_gaussian)
        self.sigma_input = Linear(hidden_size, num_gaussian)
        self.output = Linear(num_gaussian, hidden_size)

    def forward(self, x):
        mu = self.mu_input(x)
        sigma = self.sigma_input(x)
        rand_tensor = mu + normal_random_like(mu) * torch.exp(sigma)
        return self.output(rand_tensor), (mu, sigma)

    def generate(self, n_sentenses):
        shape = (n_sentenses, self.num_gaussian)
        mu = torch.zeros(shape).to(DEVICE)
        sigma = torch.ones(shape).to(DEVICE)
        rand_tensor = torch.normal(mu, sigma)
        return self.output(rand_tensor)


# %%
class LayerwiseVAE(VAEModule):
    def __init__(
        self,
        hidden_size,
        num_layers,
        num_gaussian,
    ) -> None:
        super().__init__()
        self.num_gaussian = num_gaussian
        self.mu_layer = ModuleList([Linear(hidden_size, num_gaussian) for _ in range(num_layers)])
        self.sigma_layer = ModuleList([Linear(hidden_size, num_gaussian) for _ in range(num_layers)])
        self.output_layer = ModuleList([Linear(num_gaussian, hidden_size) for _ in range(num_layers)])
   
    def forward(self, hidden):
        mu = [layer(x) for layer, x in zip(self.mu_layer, hidden)]
        sigma = [layer(x) for layer, x in zip(self.sigma_layer, hidden)]
        random = [mu + self.normal_random_like(mu) * torch.exp(sigma) for mu, sigma in zip(mu, sigma)]
        output = [layer(x) for layer, x in zip(self.output_layer, random)]
        return torch.stack(output), (torch.stack(mu), torch.stack(sigma))

    @staticmethod
    def normal_random_like(tensor):
        return torch.normal(torch.zeros_like(tensor), torch.ones_like(tensor))
    
    def generate(self, n_sentenses):
        shape = (n_sentenses, self.num_gaussian)
        mu = torch.zeros(shape).to(DEVICE)
        sigma = torch.ones(shape).to(DEVICE)
        return torch.stack([
            layer(torch.normal(mu, sigma)) for layer in self.output_layer
        ])

# %%
class LSTMVAE(VAEModule):
    def __init__(
        self,
        n_dim,
        n_layers,
        n_gaussian,
    ) -> None:
        super().__init__()
        self.encoder = LSTM(
            input_size=n_dim,
            hidden_size=n_dim, 
            num_layers=n_layers, 
            batch_first=True
        )
        set_lstm_spectral_norm(self.encoder)
        self.hidden_vae = LayerwiseVAE(
            hidden_size=n_dim,
            num_layers=n_layers,
            num_gaussian=n_gaussian
        )
        self.cell_vae = LayerwiseVAE(
            hidden_size=n_dim,
            num_layers=n_layers,
            num_gaussian=n_gaussian
        )
        self.decoder = LSTMDecoder(
            hidden_size=n_dim,
            num_layers=n_layers,
        )
        self.initial = Parameter(torch.zeros([1, n_dim]))
    
    def forward(self, tensor):
        _, (h, c) = self.encoder(tensor)
        h, (h_mu, h_sigma) = self.hidden_vae(h)
        c, (c_mu, c_sigma) = self.cell_vae(c)
        initial_tensor = torch.stack([self.initial] * tensor.shape[0])
        output = self.decoder(initial_tensor, (h, c), tensor.shape[1])
        return output, (torch.cat([h_mu, c_mu]), torch.cat([h_sigma, c_sigma]))

    def generate(self, n_sentenses, length):
        h = self.hidden_vae.generate(n_sentenses)
        c = self.cell_vae.generate(n_sentenses)
        initial_tensor = torch.stack([self.initial] * n_sentenses)
        return self.decoder(initial_tensor, (h, c), length)


class ResidualBlock(Module):
    def __init__(self, model: Module):
        super().__init__()
        self.model = model
        self.param = Parameter(torch.rand(1))

    def forward(self, x, *args, **kwargs):
        return x * self.param + self.model(x, *args, **kwargs)


class TranformerVAE(Module):
    def __init__(
        self,
        n_dim: int,
        n_layers: int,
        n_head: int,
        n_gaussian: int,
        max_len: int,
    ):
        super().__init__()
        self.pos_enc_emb = Parameter(torch.rand((max_len, n_dim)))
        self.pos_dec_emb = Parameter(torch.rand((max_len, n_dim)))
        self.n_dim = n_dim
        self.max_len = max_len
        self.encoder = TransformerEncoder(
            ResidualBlock(TransformerEncoderLayer(n_dim, n_head)),
            num_layers=n_layers
        )
        self.vae = VAE(n_dim, n_gaussian)
        self.decoder = TransformerEncoder(
            ResidualBlock(TransformerEncoderLayer(n_dim, n_head)),
            num_layers=n_layers
        )
    
    def forward(self, x):
        # key_padding_mask
        mask = (x == 0).all(axis=-1)
        x = (x + self.pos_enc_emb).transpose(0, 1)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.transpose(0, 1)
        x, mu_sigma = self.vae(x)
        x = (x + self.pos_dec_emb).transpose(0, 1)
        x = self.decoder(x, src_key_padding_mask=mask)
        return x.transpose(0, 1), mu_sigma
    
    def generate(self, n_sentences, length):
        x = torch.stack([self.vae.generate(n_sentences) for _ in range(length)], dim=1)
        x = (x + self.pos_dec_emb[:length]).transpose(0, 1)
        return self.decoder(x).transpose(0, 1)

# %%
class EmbWrapper(Module):
    def __init__(
        self,
        model, 
        n_words, 
        n_dim,
    ) -> None:
        super().__init__()
        self.model = model
        self.emb_layer = Embedding(n_words + 1, n_dim, padding_idx=0)
        self.output_layer = Linear(n_dim, out_features=n_words + 1)

    def forward(self, tensor):
        x = self.emb_layer(tensor)
        x, other_outputs = self.model(x)
        return self.output_layer(x), other_outputs
    
    def generate(self, n_sentences, length):
        x = self.model.generate(n_sentences, length)
        return self.output_layer(x).argmax(axis=-1)


def make_lstm(
    n_dim,
    n_layers,
    n_gaussian,
    n_words,
):
    vae_model = LSTMVAE(n_dim, n_layers, n_gaussian)
    return EmbWrapper(vae_model, n_words, n_dim)


def make_transformer(
    n_dim,
    n_layers,
    n_head,
    n_gaussian,
    max_len,
    n_words,
):
    model = TranformerVAE(n_dim, n_layers, n_head, n_gaussian, max_len)
    return EmbWrapper(model, n_words, n_dim)

# %%
