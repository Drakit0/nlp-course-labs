import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F

try:
    from decoder import TransformerDecoder
    from encoder import TransformerEncoder
except ModuleNotFoundError:
    from src.decoder import TransformerDecoder
    from src.encoder import TransformerEncoder

class Transformer(nn.Module):

    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, max_enc_position_embeddings: int, max_dec_position_embeddings: int,
                enc_d_model: int, dec_d_model: int, num_attention_heads: int, enc_intermediate_size: int, 
                dec_intermediate_size: int, num_enc_hidden_layers: int, num_dec_hidden_layers: int
                ):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_vocab_size, max_enc_position_embeddings, enc_d_model, num_attention_heads, enc_intermediate_size, num_enc_hidden_layers)
        self.decoder = TransformerDecoder(tgt_vocab_size, max_dec_position_embeddings, dec_d_model, num_attention_heads, dec_intermediate_size, num_dec_hidden_layers)
        self.output_linear = nn.Linear(dec_d_model, tgt_vocab_size)

    def forward(self, src_input: torch.Tensor, tgt_input: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        enc_output = self.encoder(src_input, attn_mask)

        dec_output = self.decoder(tgt_input, enc_output)

        dec_output = self.output_linear(dec_output)

        return dec_output
    
    def generate(self, src_input: torch.Tensor, max_length: int = 50, decoding_strategy: str = 'greedy', **kwargs) -> torch.Tensor:
        if decoding_strategy == 'greedy':
            return self.__greedy_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'beam_search':
            return self.__beam_search_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'sampling':
            return self.__sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'top_k':
            return self.__top_k_sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'top_p':
            return self.__top_p_sampling_decode(src_input, max_length, **kwargs)
        elif decoding_strategy == 'contrastive':
            return self.__contrastive_decode(src_input, max_length, **kwargs)
        
        else:
            raise ValueError(f"Invalid decoding strategy: {decoding_strategy}")

    def __greedy_decode(self, src_input: torch.Tensor, max_length: int, **kwargs) -> torch.Tensor:
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)  # Default SOS token index
        EOS_token = kwargs.get('EOS_token', 3)  # Default EOS token index

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_output = self.decoder(tgt_input, enc_output)
            dec_output = self.output_linear(dec_output)
            logits = dec_output[:, -1, :]  # Shape: (batch_size, vocab_size)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Shape: (batch_size, 1)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if (next_token == EOS_token).all():
                break

        generated_sequence = tgt_input[:, 1:]  # Shape: (batch_size, seq_len)
        return generated_sequence

    def __beam_search_decode(self, src_input: torch.Tensor, max_length: int, beam_size: int = 3, **kwargs) -> torch.Tensor:
        batch_size = src_input.size(0)
        if batch_size != 1:
            raise NotImplementedError("Beam search decoding currently only supports batch_size=1")
        device = src_input.device

        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((1, 1), SOS_token, dtype=torch.long, device=device)
        beam = [(tgt_input, 0)]  # Each item is (sequence tensor, cumulative log probability)

        for _ in range(max_length):
            candidates = []
            for seq, score in beam:
                if seq[0, -1].item() == EOS_token:
                    candidates.append((seq, score))
                else:
                    dec_output = self.decoder(seq, enc_output)
                    dec_output = self.output_linear(dec_output)
                    logits = dec_output[:, -1, :]  # Shape: (1, vocab_size)
                    log_probs = torch.log_softmax(logits, dim=-1)  # Shape: (1, vocab_size)
                    for next_token in range(log_probs.size(1)):
                        new_seq = torch.cat([seq, torch.tensor([[next_token]], device=seq.device)], dim=1)  # Shape: (1, seq_len+1)
                        new_score = score + log_probs[0, next_token].item()
                        candidates.append((new_seq, new_score))
            beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            if all(seq[0, -1].item() == EOS_token for seq, _ in beam):
                break
        best_seq = beam[0][0]
        generated_sequence = best_seq[:, 1:]  # Shape: (1, seq_len)
        return generated_sequence
    
    def __sampling_decode(self, src_input: torch.Tensor, max_length: int, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_output = self.decoder(tgt_input, enc_output)
            dec_output = self.output_linear(dec_output)
            logits = dec_output[:, -1, :]  # Shape: (batch_size, vocab_size)

            scaled_logits = logits / temperature

            probs = torch.softmax(scaled_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)  # Shape: (batch_size, 1)

            tgt_input = torch.cat([tgt_input, next_token], dim=1)

            if (next_token == EOS_token).all():
                break

        generated_sequence = tgt_input[:, 1:]  # Shape: (batch_size, generated_seq_len)
        return generated_sequence

    def __top_k_sampling_decode(self, src_input: torch.Tensor, max_length: int, k: int = 10, **kwargs) -> torch.Tensor:
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_output = self.decoder(tgt_input, enc_output)
            dec_output = self.output_linear(dec_output)
            logits = dec_output[:, -1, :] # Shape: (batch_size, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)
            topk_log_probs, topk_indices = torch.topk(log_probs, k, dim=-1)
            probs = torch.softmax(topk_log_probs, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # Shape: (batch_size, 1)
            next_token = topk_indices.gather(-1, next_token)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if (next_token == EOS_token).all():
                break

        generated_sequence = tgt_input[:, 1:]
        return generated_sequence
    

    def __top_p_sampling_decode(self, src_input: torch.Tensor, max_length: int, p: float = 0.9, **kwargs) -> torch.Tensor:
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_output = self.decoder(tgt_input, enc_output)
            dec_output = self.output_linear(dec_output)
            logits = dec_output[:, -1, :]  # Shape: (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > p
            sorted_probs[sorted_indices_to_remove] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
            if (next_token == EOS_token).all():
                break

        generated_sequence = tgt_input[:, 1:]
        return generated_sequence
    
    def __contrastive_decode(self, src_input: torch.Tensor, max_length: int, k: int = 5, alpha: float = 0.6, **kwargs) -> torch.Tensor:
        attn_mask = kwargs.get('attn_mask', None)
        enc_output = self.encoder(src_input, attn_mask)

        batch_size = src_input.size(0)
        device = src_input.device

        SOS_token = kwargs.get('SOS_token', 2)
        EOS_token = kwargs.get('EOS_token', 3)

        tgt_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=device)

        for _ in range(max_length):
            dec_output = self.decoder(tgt_input, enc_output)
            dec_output = self.output_linear(dec_output)
            logits = dec_output[:, -1, :]  # Shape: (batch_size, vocab_size)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

            expanded_tgt_input = tgt_input.repeat(k, 1)  # Shape: (k, seq_len)
            next_tokens = topk_indices.squeeze(0).unsqueeze(-1)  # Shape: (k, 1)
            y_candidates = torch.cat([expanded_tgt_input, next_tokens], dim=1)  # Shape: (k, seq_len + 1)

            dec_outputs_candidate = self.decoder(y_candidates, enc_output.repeat(k, 1, 1))

            h_v = dec_outputs_candidate[:, -1, :]  # Shape: (k, hidden_size)
            h_j = dec_outputs_candidate[:, :-1, :]  # Shape: (k, seq_len, hidden_size)

            h_v_norm = F.normalize(h_v, p=2, dim=-1)  # Shape: (k, hidden_size)
            h_j_norm = F.normalize(h_j, p=2, dim=-1)  # Shape: (k, seq_len, hidden_size)

            cos_sim = torch.bmm(h_v_norm.unsqueeze(1), h_j_norm.transpose(1, 2)).squeeze(1)  # Shape: (k, seq_len)

            max_sim = cos_sim.max(dim=1)[0]  # Shape: (k,)

            P_LM_v = topk_probs.squeeze(0)  # Shape: (k,)
            scores = alpha * P_LM_v - (1 - alpha) * max_sim  # Shape: (k,)

            best_idx = scores.argmax()
            best_token = next_tokens[best_idx].unsqueeze(0)  # Shape: (1, 1)
            tgt_input = torch.cat([tgt_input, best_token], dim=1)  # Shape: (1, seq_len + 1)

            if best_token.item() == EOS_token:
                break

        generated_sequence = tgt_input[:, 1:]
        return generated_sequence


    
if __name__ == "__main__":
    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 100
    max_position_embeddings = 128
    intermediate_size = 64

    transformer = Transformer(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, max_enc_position_embeddings=max_position_embeddings, 
                            max_dec_position_embeddings=max_position_embeddings, enc_d_model=d_model, dec_d_model=d_model, 
                            num_attention_heads=num_heads, enc_intermediate_size=intermediate_size, dec_intermediate_size=intermediate_size, 
                            num_enc_hidden_layers=num_layers, num_dec_hidden_layers=num_layers)

    src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(100):
        optimizer.zero_grad()
        output = transformer(src_data, tgt_data[:, :-1], attn_mask=None)
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    transformer.eval()

    val_src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
    val_tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

    with torch.no_grad():

        val_output = transformer(val_src_data, val_tgt_data[:, :-1])
        val_loss = criterion(val_output.contiguous().view(-1, tgt_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
        print(f"Validation Loss: {val_loss.item()}")
