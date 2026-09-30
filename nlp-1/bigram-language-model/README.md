# Bigram language model

Lab 2 of NLP I. A character-level bigram model trained on a list of Spanish names, used to generate new names and to score words.

## Implemented

`src/data_processing.py`
- `load_and_preprocess_data`: reads the names file, adds start and end tokens and builds the bigrams.
- `char_to_index`, `index_to_char`: character lookup tables including the start and end tokens.
- `count_bigrams`: bigram count matrix as a torch tensor.

`src/bigram_model.py`
- `bigrams_count_to_probabilities`: row normalisation of the count matrix with additive smoothing.
- `sample_next_character` and `generate_name`: sampling a name character by character until the end token.
- `calculate_log_likelihood` and `calculate_neg_mean_log_likelihood`: scoring a word and a list of words.

## Run

The names file from the course is not included. Install the requirements and import the modules from `src/`:

```
pip install -r requirements.txt
cd src
python -c "import bigram_model"
```
