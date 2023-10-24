from typing import Dict, Any
import torch.nn as nn
from src.benchmark.model import Model

from torch.utils.data import Dataset, DataLoader

class MLP(Model):
    """MLP model."""
        
    def __init__(self, name, **kwargs):
        """Initialize the model."""
        super().__init__(name, **kwargs)
        self.model = nn.Sequential()

        # Model parameters
        self.input_dim = kwargs['input_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.num_layers = kwargs['num_layers']
        self.dropout = kwargs['dropout']
        self.dropout_prob = kwargs['dropout_prob']

        # Construct model
        self._construct_model()

    def forward(self, x):
        """ Forward pass. """
        return self.model(x)

    def _construct_model(self,) -> None:
        """Construct the model."""
        for layer in range(self.num_layers):
            if layer == 0:
                self.model.add_module(f'fc{layer}', nn.Linear(self.input_dim, self.hidden_dim[layer]))
                self.model.add_module(f'act{layer}', nn.ReLU())
            else:
                self.model.add_module(f'fc{layer}', nn.Linear(self.hidden_dim[layer-1], self.hidden_dim[layer]))
                self.model.add_module(f'act{layer}', nn.ReLU())

        self.model.add_module(f'fc{self.num_layers}', nn.Linear(self.hidden_dim[-1], 1))
        self.model.add_module(f'act{self.num_layers}', nn.Sigmoid())

        if self.dropout:
            self.model.add_module(f'dropout{self.num_layers}', nn.Dropout(self.dropout_prob))

    def train(self, train_loader = None, **kwargs):
        """Train the model."""
        if train_loader is None:
            raise ValueError("train_loader is None.")
        
    
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