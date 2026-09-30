import torch
from torch import nn
import torch.optim as optim

class SkipGramNeg(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, noise_dist: torch.Tensor = None):
        super().__init__()
        self.n_vocab: int = n_vocab
        self.n_embed: int = n_embed
        self.noise_dist: torch.Tensor = noise_dist

        self.in_embed: nn.Embedding = nn.Embedding(n_vocab, n_embed)
        self.out_embed: nn.Embedding = nn.Embedding(n_vocab, n_embed)

        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward_input(self, input_words: torch.Tensor) -> torch.Tensor:
        input_vectors: torch.Tensor = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words: torch.Tensor) -> torch.Tensor:
        output_vectors: torch.Tensor = self.out_embed(output_words)
        return output_vectors

    def forward_noise(self, batch_size: int, n_samples: int) -> torch.Tensor:
        if self.noise_dist is None:
            noise_dist: torch.Tensor = torch.ones(self.n_vocab)
        else:
            noise_dist: torch.Tensor = self.noise_dist

        noise_words: torch.Tensor = torch.multinomial(noise_dist, batch_size * n_samples, replacement=True)

        device: str = "cuda" if self.out_embed.weight.is_cuda else "cpu"
        noise_words: torch.Tensor = noise_words.to(device)

        noise_vectors: torch.Tensor = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)

        return noise_vectors

    
class NegativeSamplingLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input_vectors: torch.Tensor, output_vectors: torch.Tensor,
                noise_vectors: torch.Tensor) -> torch.Tensor:

        out_loss = torch.log(torch.sigmoid(output_vectors @ input_vectors.T)).sum()
        noise_loss = torch.log(torch.sigmoid(-noise_vectors @ input_vectors.T)).sum()
        return -1*torch.mean((out_loss + noise_loss))