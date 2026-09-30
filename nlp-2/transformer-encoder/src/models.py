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

    def scaled_dot_product_attention(self, q, k, v):

        dim_k = self.wk.out_features

        scores = (q.matmul(k.transpose(1, 2)))/math.sqrt(dim_k)

        weights = F.softmax(scores, 2)

        output = weights.matmul(v)

        return output, weights

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        output, _ = self.scaled_dot_product_attention(q, k, v)

        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model, 
                                                  int(d_model/num_attention_heads), 
                                                  int(d_model/num_attention_heads), 
                                                  int(d_model/num_attention_heads)) for _ in range(num_attention_heads)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, hidden_state):
        x = self.output_linear(torch.concat([attention_head.forward(hidden_state) for attention_head in self.heads], 2))
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

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward = FeedForward(d_model, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_state = self.attention(self.layer_norm_1(x))
        
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
    
class TransformerEncoder(nn.Module):

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                num_attention_heads: int, intermediate_size: int, num_hidden_layers: int
                 ):
        super(TransformerEncoder, self).__init__()
        self.embeddings = Embeddings(vocab_size, max_position_embeddings, d_model)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_attention_heads, intermediate_size) for _ in range(num_hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(x)
        for attention_layer in self.layers:
            x = attention_layer(x)
        return x
    
class ClassificationHead(nn.Module):

    def __init__(self, d_model: int, num_classes: int, dropout_prob: float):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.linear(x))
        return x
    
class TransformerForSequenceClassification(nn.Module):

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                num_attention_heads: int, intermediate_size: int, num_hidden_layers: int,
                num_classes: int, dropout_prob: float):
        super(TransformerForSequenceClassification, self).__init__()
        self.transformer_encoder = TransformerEncoder(vocab_size, 
                                                      max_position_embeddings, d_model, 
                                                      num_attention_heads, 
                                                      intermediate_size, 
                                                      num_hidden_layers)
        self.classifier = ClassificationHead(d_model, num_classes, dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.transformer_encoder(input_ids)

        x = x[:, 0, :]
        
        x = self.classifier(x)
        return x
