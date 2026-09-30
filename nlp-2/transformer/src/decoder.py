import torch
import torch.nn as nn

try:
    from utils import MultiHeadAttention, FeedForward, Embeddings
except ModuleNotFoundError:
    from src.utils import MultiHeadAttention, FeedForward, Embeddings

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerDecoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_attention_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        hidden_state = self.self_attention(self.layer_norm_1(x), self.layer_norm_1(x), self.layer_norm_1(x), tgt_mask)

        hidden_state = self.cross_attention(self.layer_norm_2(x), enc_output, enc_output)
        
        x = self.feed_forward(self.layer_norm_3(hidden_state))

        return x

class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                num_attention_heads: int, intermediate_size: int, num_hidden_layers: int):
        super(TransformerDecoder, self).__init__()
        self.embeddings = Embeddings(vocab_size, max_position_embeddings, d_model)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_attention_heads, intermediate_size) for _ in range(num_hidden_layers)])

    def forward(self, input_ids: torch.Tensor, enc_output: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_ids)
        batch_size, seq_len, _ = x.size()

        inverted_mask = torch.tril(torch.ones((x.shape[-2], x.shape[-2])))
        tgt_mask = inverted_mask*-1 + torch.ones((x.shape[-2], x.shape[-2]))

        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask)

        return x

