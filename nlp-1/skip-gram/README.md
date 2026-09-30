# Skip-gram with negative sampling

Lab 4 of NLP I. Word embeddings trained with the skip-gram objective and negative sampling in PyTorch.

## Implemented

`src/data_processing.py`
- `load_and_preprocess_data`: reads and tokenizes the corpus.
- `create_lookup_tables`, `subsample_words` (frequent word subsampling), `get_target` (context window), `get_batches`, and `cosine_similarity` for nearest-neighbour checks.

`src/skipgram.py`
- `SkipGramNeg`: input and output embedding tables, `forward_input`, `forward_output` and `forward_noise` (negative samples drawn from a noise distribution).
- `NegativeSamplingLoss`.

`src/train.py`: `train_skipgram`, the training loop.

`plot_study.txt` is my short note on one embedding plot. It is a single observation about a few word pairs, not an evaluation.

## Run

The text8 corpus is not included. Install the requirements and import the modules from `src/`:

```
pip install -r requirements.txt
cd src
python -c "import skipgram, train"
```
