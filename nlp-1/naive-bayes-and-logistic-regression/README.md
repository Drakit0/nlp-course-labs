# Naive Bayes and logistic regression

Lab 3 of NLP I. Two binary sentiment classifiers on bag-of-words features, both written with torch tensors.

## Implemented

`src/data_processing.py`
- `read_sentiment_examples`: reads a tab-separated file (sentence, label) into `SentimentExample` objects.
- `build_vocab` and `bag_of_words`: vocabulary from the training examples and count vectors.

`src/naive_bayes.py`, class `NaiveBayes`
- Class priors, conditional word probabilities with additive smoothing (`delta`), class posteriors, `predict` and `predict_proba`.

`src/logistic_regression.py`, class `LogisticRegression`
- Parameter initialisation, sigmoid, binary cross-entropy loss, gradient descent in `fit`, `predict` with a cutoff and `predict_proba`.

`src/utils.py`: `evaluate_classification`.

## Run

The train and test files belong to the course and are not included. Install the requirements and import the modules from `src/`:

```
pip install -r requirements.txt
cd src
python -c "import naive_bayes, logistic_regression"
```
