import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1] # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)

        return self.conditional_probabilities

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        print(labels)
        class_priors: Dict[int, torch.Tensor] = {}
        for label in labels:
            if label.item() in class_priors.keys():
                class_priors[label.item()] += torch.tensor(1)
                
            else:
                class_priors[label.item()] = torch.tensor(1)
                
        class_priors: Dict[int, torch.Tensor] = {key: torch.tensor(val.item()/len(labels))for key, val in class_priors.items()}

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        class_word_counts: Dict[int, torch.Tensor] = {}
                
        for label in labels:
            total_count = torch.sum(features[labels == label.item()], dim=0)
            class_word_counts[label.item()] = (total_count + delta) / (torch.sum(total_count) + delta * self.vocab_size)
            
        return class_word_counts

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        log_posteriors:torch.Tensor = torch.log(torch.tensor([self.class_priors[c].item() for c in self.class_priors.keys()]))
        
        for c in self.class_priors.keys():
            log_posteriors[c] += torch.sum(torch.log(self.conditional_probabilities[c]) * feature)
            
        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        probabilities:torch.Tensor = self.predict_proba(feature)
        pred: int = torch.argmax(probabilities).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        posteriors:torch.Tensor = self.estimate_class_posteriors(feature)
        exponential_posterior_sum:torch.Tensor = torch.sum(torch.exp(posteriors))
        probs: torch.Tensor = torch.exp(posteriors) / exponential_posterior_sum
        return probs
