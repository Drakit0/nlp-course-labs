# Transformer and decoding strategies

Lab 3 of NLP II. A full encoder-decoder transformer in PyTorch with several decoding strategies.

## Implemented

`src/utils.py`: `AttentionHead` (scaled dot-product attention with optional mask), `MultiHeadAttention`, `FeedForward`, `Embeddings`.

`src/encoder.py`: `TransformerEncoderLayer`, `TransformerEncoder`.

`src/decoder.py`: `TransformerDecoderLayer` (masked self-attention and cross-attention), `TransformerDecoder`.

`src/transformer.py`: `Transformer` with `forward` and `generate`, which dispatches to these decoding methods: greedy, beam search, sampling with temperature, top-k, top-p and contrastive search.

## Run

```
pip install -r requirements.txt
cd src
python -c "import transformer"
```
