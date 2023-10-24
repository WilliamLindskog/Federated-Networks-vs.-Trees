from typing import Dict, Any
import torch.nn as nn
from src.benchmark.model import Model

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from tqdm import tqdm
import torch

class MLP(Model):
    """MLP model."""
        
    def __init__(self, name, **kwargs):
        """Initialize the model."""
        super().__init__(name, **kwargs)
        # Set task and model
        self.task = kwargs['task']
        self.model = nn.Sequential()
        self.device = kwargs['device']

        # Set model architecture
        self._set_model_architecture(**kwargs)

        # Construct model
        self._construct_model()

        # Set model parameters
        self._set_model_params(**kwargs)

    def forward(self, x):
        """ Forward pass. """
        return self.model(x).to(self.device)

    def train(self, train_loader = None, **kwargs):
        """Train the model."""
        if train_loader is None:
            raise ValueError("train_loader is None.")
        
        # Train model
        self.model.train()
        for epoch in range(self.epochs):
            for _, (data, target) in enumerate(tqdm(train_loader)):
                # Send data to device
                data, target = data, target
                target = target.unsqueeze(1)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self._loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

    
    def predict(self, test_data, **kwargs):
        """Predict the test data."""
        pass
    
    def evaluate(self, test_loader: DataLoader, **kwargs):
        """Evaluate the model."""
        # Evaluate model
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(tqdm(test_loader)):
                # Send data to device
                data, target = data, target
                target = target.unsqueeze(1)
                output = self.model(data)
                pred = torch.round(output)
                correct += (pred == target).sum().item()
                total += target.size(0)

        print(f"Accuracy: {correct/total}")
    
    def save(self, path, **kwargs):
        """Save the model."""
        pass
    
    def load(self, path, **kwargs):
        """Load the model."""
        pass

    def _set_model_architecture(self, **kwargs):
        """Set model architecture."""
        # Model architecture
        self.input_dim = kwargs['input_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.num_layers = kwargs['num_layers']
        self.dropout = kwargs['dropout']
        self.dropout_prob = kwargs['dropout_prob']

    def _set_model_params(self, **kwargs):
        """Set model parameters."""
        # Model parameters
        self.epochs = kwargs['num_epochs']
        self.lr = kwargs['learning_rate']
        self.batch_size = kwargs['batch_size']
        self.device = kwargs['device']

        # Loss function
        loss_fn = kwargs['loss_fn']
        self._set_loss_fn(loss_fn)

        # Optimizer
        optimizer = kwargs['optimizer']
        self._set_optimizer(optimizer)
        
    def _set_loss_fn(self, loss_fn: str):
        if self.task in ['binary', 'multi']:
            assert loss_fn == 'ce'
            self._loss_fn = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss function {loss_fn} not implemented.")
        
    def _set_optimizer(self, optimizer: str):
        # Optimizer
        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=self.lr,)
        else:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented.")
        
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

        self.model.to(self.device)