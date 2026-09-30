import torch

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class LogisticRegression:
    def __init__(self, random_state: int):
        self._weights: torch.Tensor = None
        self.random_state: int = random_state

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float,
        epochs: int,
    ):
        self.weights:torch.Tensor = self.initialize_parameters(features.shape[1], random_state = 42)
        
        for epoch in range(epochs):
            predict = self.predict_proba(features)
            loss = self.binary_cross_entropy_loss(predict, labels)
            
            if epoch%100 == 0:
                print(f"iter:{epoch}\tloss:{loss}")
                
            dw = torch.matmul(features.T, (predict - labels)) / labels.shape[0]
            db = (predict - labels).sum() / labels.shape[0]
            self.weights[:-1] -= learning_rate * dw
            self.weights[-1] -= learning_rate * db
        
    def predict(self, features: torch.Tensor, cutoff: float = 0.5) -> torch.Tensor:
        probabilities:torch.Tensor = self.predict_proba(features)
        decisions: torch.Tensor = (probabilities >= cutoff).int()
        return decisions

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise ValueError("Model not trained. Call the 'train' method first.")
        
        features:torch.Tensor = torch.cat((features, torch.tensor([[1] for _ in range(features.shape[0])])), dim = 1)
        probabilities: torch.Tensor = self.sigmoid(torch.sum(self.weights * features, 1))
        
        return probabilities

    def initialize_parameters(self, dim: int, random_state: int) -> torch.Tensor:
        torch.manual_seed(random_state)
        
        params: torch.Tensor = torch.normal(0, 1, (dim+1,))
        
        return params

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = torch.nn.functional.sigmoid(z)
        return result

    @staticmethod
    def binary_cross_entropy_loss(
        predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        tensor_ones:torch.Tensor = torch.ones(predictions.shape)
        ce_loss: torch.Tensor = -torch.mean(targets * torch.log(predictions) + (tensor_ones - targets) * torch.log(tensor_ones - predictions))
        return ce_loss

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights: torch.Tensor = value

