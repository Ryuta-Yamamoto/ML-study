# %%
import torch
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
)
from torch.tensor import Tensor


class VAEModule(Module):
    def generate(self, n_sentenses):
        raise NotImplementedError


# %%

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

    def forward(self, initial_input, hidden_cell_tuple, length):
        tensors = []
        previous = initial_input
        for _ in range(length):
            print(_, previous.shape, hidden_cell_tuple[0].shape)
            output, hidden_cell_tuple = self.layer(previous, hidden_cell_tuple)
            tensors.append(output)
            previous = output
        return torch.cat(tensors, axis=1)


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
        self.mu_layer = [Linear(hidden_size, num_gaussian) for _ in range(num_layers)]
        self.sigma_layer = [Linear(hidden_size, num_gaussian) for _ in range(num_layers)]
        self.output_layer = [Linear(num_gaussian, hidden_size) for _ in range(num_layers)]
   
    def forward(self, hidden):
        mu = [layer(x) for layer, x in zip(self.mu_layer, hidden)]
        sigma = [layer(x) for layer, x in zip(self.sigma_layer, hidden)]
        random = [mu + self.normal_random_like(mu) * sigma for mu, sigma in zip(mu, sigma)]
        output = [layer(x) for layer, x in zip(self.output_layer, random)]
        return torch.stack(output), (torch.stack(mu), torch.stack(sigma))

    @staticmethod
    def normal_random_like(tensor):
        return torch.normal(torch.zeros_like(tensor), torch.ones_like(tensor))
    
    def generate(self, n_sentenses):
        shape = (n_sentenses, self.num_gaussian)
        mu = torch.zeros(shape)
        sigma = torch.ones(shape)
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
        self.encorder = LSTM(
            input_size=n_dim,
            hidden_size=n_dim, 
            num_layers=n_layers, 
            batch_first=True
        )
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
        _, (h, c) = self.encorder(tensor)
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



# %%
