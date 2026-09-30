
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionHead(nn.Module):

    def __init__(self, d_model: int, d_k: int, d_q: int, d_v: int):
        super(AttentionHead, self).__init__()

        self.wq = nn.Linear(d_model, d_q)
        self.wk = nn.Linear(d_model, d_k)
        self.wv = nn.Linear(d_model, d_v)

    def scaled_dot_product_attention(self, q, k, v, mask=None):

        dim_k = int(k.shape[-1])

        scores = (q.matmul(k.transpose(1, 2)))/math.sqrt(dim_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, value=float('-inf'))

        weights = F.softmax(scores, 2)

        output = weights.matmul(v)

        return output, weights

    def forward(self, x_q, x_k, x_v, mask=None):
        q = self.wq(x_q)
        k = self.wk(x_k)
        v = self.wv(x_v)

        output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_attention_heads == 0, "d_model must be divisible by num_attention_heads"
        
        self.heads = nn.ModuleList([AttentionHead(d_model, 
                                            int(d_model/num_attention_heads), 
                                            int(d_model/num_attention_heads), 
                                            int(d_model/num_attention_heads),) for _ in range(num_attention_heads)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_k, x_v, mask=None):
        x = self.output_linear(torch.concat([attention_head.forward(x_q, x_k, x_v, mask) for attention_head in self.heads], 2))
        return x
    
class FeedForward(nn.Module):

    def __init__(self, d_model: int, intermediate_size: int):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, d_model)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_2(self.gelu(self.linear_1(x)))
        return x
    
class Embeddings(nn.Module):

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)

        return embeddings

