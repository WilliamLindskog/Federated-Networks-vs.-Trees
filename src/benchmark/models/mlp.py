from typing import Dict, Any
import torch.nn as nn
from src.benchmark.model import Model

from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from tqdm import tqdm
import torch

class MLP(Model):
    """MLP model."""
        
    def __init__(self, **kwargs):
        """Initialize the model."""
        super(Model, self).__init__()
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
    
    def _one_train_epoch(self, data, target, **kwargs):
        """One train epoch."""
        # target = target.flatten()
        self.optimizer.zero_grad()
        output = self.model(data)
        target = target.to(self.device)
        loss = self._loss_fn(output, target)
        loss.backward()
        self.optimizer.step()

        return loss, output

    def train(self, train_loader = None, **kwargs):
        """Train the model."""
        if train_loader is None:
            raise ValueError("train_loader is None.")
        
        # Train model
        self.model.train()
        correct, total = 0, 0
        loss = 0.0
        for epoch in range(self.epochs):
            for _, (data, target) in enumerate(train_loader):
                # one train step
                loss, output = self._one_train_epoch(data, target.long())

                if self.task in ['binary', 'multi']:
                    pred = torch.round(output)
                    # correct += (pred == target).sum().item()
                    total += target.size(0)
                    correct += (torch.max(output.data, 1)[1] == target).sum().item()
                else: 
                    raise NotImplementedError(f"Task {self.task} not implemented.")

            if self.task in ['binary', 'multi']:
                metric, metric_name = correct/total, 'accuracy'
                # print(f'Epoch: {epoch+1}/{self.epochs}, Loss: {loss.item()}, {metric_name}: {metric}')
            else:
                raise NotImplementedError(f"Task {self.task} not implemented.")
            
        return {f'{metric_name}': metric}
    
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
            for _, (data, target) in enumerate(test_loader):
                # Send data to device
                data, target = data, target
                target = target.long()
                output = self.model(data)
                correct += (torch.max(output.data, 1)[1] == target).sum().item()
                total += target.size(0)

        loss = self._loss_fn(output, target)

        # return loss and accuracy
        return loss, correct / total
    
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
        self.output_dim = kwargs['output_dim']
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
        if self.task == 'binary':
            assert loss_fn == 'bce'
            self._loss_fn = nn.CrossEntropyLoss()
        elif self.task == 'multi':
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

        #if self.task == 'binary': 
        #    self.model.add_module(f'fc{self.num_layers}', nn.Linear(self.hidden_dim[-1], 1))
        #elif self.task == 'multi':
        self.model.add_module(f'fc{self.num_layers}', nn.Linear(self.hidden_dim[-1], self.output_dim))
        #else:
        #    raise NotImplementedError(f"Task {self.task} not implemented.")
        #self.model.add_module(f'act{self.num_layers}', nn.Sigmoid())

        #if self.dropout:
        #    self.model.add_module(f'dropout{self.num_layers}', nn.Dropout(self.dropout_prob))

        self.model.to(self.device)