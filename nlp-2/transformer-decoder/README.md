# Transformer decoder

Lab 2 of NLP II. A decoder-only, character-level transformer trained on a list of Spanish names, used to generate new names.

## Implemented

`src/decoder.py`
- `AttentionHead` and `MultiHeadAttention` with a causal mask, `FeedForward`, `TransformerDecoderLayer`.
- `Embeddings`, `TransformerDecoder`, and `TransformerForLanguageModeling` (decoder plus output projection).

`src/data_processing.py`
- `load_and_preprocess_data`, `CharTokenizer` (`encode`, `decode`), `NameDataset` and `collate_fn`.

`src/train.py`: `train`, the training loop with validation loss.

## Run

The names file belongs to the course and is not included. Install the requirements and import the modules from `src/`:

```
pip install -r requirements.txt
cd src
python -c "import decoder"
```
