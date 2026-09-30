# Transformer encoder

Lab 1 of NLP II. A transformer encoder for sequence classification, written from scratch in PyTorch.

## Implemented

All in `src/models.py`:
- `AttentionHead` with scaled dot-product attention, and `MultiHeadAttention`.
- `FeedForward` and `TransformerEncoderLayer`.
- `Embeddings` (token and position) and `TransformerEncoder`, a stack of encoder layers.
- `ClassificationHead` and `TransformerForSequenceClassification`.

## Run

There is no training script in this lab. The classes are meant to be imported:

```
pip install -r requirements.txt
cd src
python -c "import models"
```
