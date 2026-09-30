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
        dim_k = self.wk.out_features

        scores = torch.Tensor((q.matmul(k.transpose(1, 2)))/math.sqrt(dim_k))
        
        if mask != None:
            scores = scores.masked_fill(mask = mask.bool(), value = float('-inf'))

        weights = F.softmax(scores, 2)

        output = weights.matmul(v)

        return output, weights

    def forward(self, x, mask=None):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadAttention, self).__init__()
        
        self.heads = nn.ModuleList([AttentionHead(d_model, 
                                            int(d_model/num_attention_heads), 
                                            int(d_model/num_attention_heads), 
                                            int(d_model/num_attention_heads),) for _ in range(num_attention_heads)])
        self.output_linear = nn.Linear(d_model, d_model)


    def forward(self, hidden_state, mask=None):
        x = self.output_linear(torch.concat([attention_head.forward(hidden_state, mask) for attention_head in self.heads], 2))
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

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerDecoderLayer, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden_state = self.self_attention(self.layer_norm_1(x), mask)
        
        x = self.feed_forward(self.layer_norm_2(hidden_state))

        return x

class Embeddings(nn.Module):

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int):
        super(Embeddings, self).__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_length = input_ids.shape[1]
        position_ids = torch.tensor([[i for i in range(seq_length)] for _ in range(input_ids.shape[0])])

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = self.layer_norm(token_embeddings + position_embeddings)

        return embeddings

class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                 num_attention_heads: int, intermediate_size: int, num_hidden_layers: int):
        super(TransformerDecoder, self).__init__()
        self.embeddings = Embeddings(vocab_size, max_position_embeddings, d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_attention_heads, intermediate_size) for _ in range(num_hidden_layers)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        
        inverted_mask = torch.tril(torch.ones((x.shape[-2], x.shape[-2])))
        mask = inverted_mask*-1 + torch.ones((x.shape[-2], x.shape[-2]))
        
        
        for attention_layer in self.layers:
            x = attention_layer(x, mask)
            
        return x

class TransformerForLanguageModeling(nn.Module):

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                 num_attention_heads: int, intermediate_size: int, num_hidden_layers: int):
        super(TransformerForLanguageModeling, self).__init__()
        self.transformer_decoder = TransformerDecoder(vocab_size, 
                                                      max_position_embeddings, 
                                                      d_model, 
                                                      num_attention_heads, 
                                                      intermediate_size, 
                                                      num_hidden_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        logits = self.lm_head(self.transformer_decoder(input_ids))
        return logits
