from src.benchmark.model import Model

class MLP(Model):
    """MLP model."""
        
    def __init__(self, name, **kwargs):
        """Initialize the model."""
        super().__init__(name, **kwargs)
        
        self.model = None
    
    def train(self, train_data, train_labels, **kwargs):
        """Train the model."""
        pass
    
    def predict(self, test_data, **kwargs):
        """Predict the test data."""
        pass
    
    def evaluate(self, test_data, test_labels, **kwargs):
        """Evaluate the model."""
        pass
    
    def save(self, path, **kwargs):
        """Save the model."""
        pass
    
    def load(self, path, **kwargs):
        """Load the model."""
        pass