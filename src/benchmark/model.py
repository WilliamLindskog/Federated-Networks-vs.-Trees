from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract class for models."""
    
    @abstractmethod
    def __init__(self, name, **kwargs):
        """Initialize the model."""
        self.name = name

    @abstractmethod
    def train(self, train_data, train_labels, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, test_data, **kwargs):
        """Predict the test data."""
        pass

    @abstractmethod
    def evaluate(self, test_data, test_labels, **kwargs):
        """Evaluate the model."""
        pass

    @abstractmethod
    def save(self, path, **kwargs):
        """Save the model."""
        pass

    @abstractmethod
    def load(self, path, **kwargs):
        """Load the model."""
        pass