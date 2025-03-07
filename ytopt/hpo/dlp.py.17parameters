import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Random seed for reproducibility
random_seed = 123
np.random.seed(random_seed)

def get_u(x, p):
    f_x2 = norm.pdf(x[0], loc=p, scale=2)
    f_x1 = norm.pdf(x[1], loc=3 * x[0] + 2, scale=2)
    u = f_x1 * f_x2
    return u

def sample_u(p):
    x1 = np.random.normal(p, 2, size=1)
    x2 = np.random.normal(loc=3 * x1 + 2, scale=2, size=1)
    return [x1[0], x2[0]]

# Generate samples and calculate PDFs
p = 2
samples = np.array([sample_u(p) for _ in range(100000)])
x1_samples, x2_samples = samples[:, 0], samples[:, 1]
train_pdfs = np.array([get_u(samples[i], p) for i in range(samples.shape[0])])

# Define the neural network
class NN_model(nn.Module):
    def __init__(self, input_dim, layer_config, activation_functions, dropout_rate):
        super(NN_model, self).__init__()
        layers = []
        prev_dim = input_dim

        # Activation map
        activation_map = {
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "ELU": nn.ELU(),
            "SiLU": nn.SiLU(),
            "softmax": nn.Softmax(dim=-1),
        }

        for num_nodes, activation_name in zip(layer_config, activation_functions):
            layers.append(nn.Linear(prev_dim, num_nodes))
            layers.append(activation_map[activation_name])
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = num_nodes

        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Xavier and He initialization
weight_init = "#P6 "  # Options: "xavier", "he", "uniform"
def weights_init(m):
    if isinstance(m, nn.Linear):
        if weight_init == "xavier":
            nn.init.xavier_uniform_(m.weight)
        elif weight_init == "he":
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif weight_init == "uniform":
            nn.init.uniform_(m.weight, -0.1, 0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

# Training function
def train_NN_pdf_via_score(model, dataloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            batch_events, batch_pdfs = batch
            batch_events = batch_events.requires_grad_(True)

            # First-order gradient
            def compute_first_order_grad(batch_events):
                est_pdf = model(batch_events).sum()
                return torch.autograd.grad(outputs=est_pdf, inputs=batch_events, create_graph=True)[0]

            # Diagonal of Hessian
            def compute_hessian_diagonal(batch_events):
                first_order_grad = compute_first_order_grad(batch_events)
                hessian_diag = []
                for i in range(batch_events.shape[1]):
                    grad2 = torch.autograd.grad(
                        outputs=first_order_grad[:, i].sum(),
                        inputs=batch_events,
                        retain_graph=True,
                        create_graph=True
                    )[0][:, i]
                    hessian_diag.append(grad2)
                return torch.stack(hessian_diag, dim=1)

            first_order_grad = compute_first_order_grad(batch_events)
            second_order_grad = compute_hessian_diagonal(batch_events)

            loss = torch.mean(torch.sum(second_order_grad + 0.5 * (first_order_grad ** 2), axis=1))
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}")

# Model hyperparameters
input_dim = 2
layer_config = [#P12 , #P13 , #P14 , #P15 , #P16 ]  # Number of nodes in each layer
activation_functions = ["#P7 ", "#P8 ", "#P9 ", "#P10 ", "#P11 "]  # Activation functions for each layer
dropout_rate = #P3 

# Training parameters
num_epochs = #P1 
batch_size = #P0 
learning_rate = #P2 
weight_decay = #P5   # L2 regularization

# Initialize model
model = NN_model(input_dim, layer_config, activation_functions, dropout_rate)
model.apply(weights_init)

# Define optimizer
optimizer = optim.#P4 (model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Create DataLoader
train_dataset = TensorDataset(torch.Tensor(samples), torch.Tensor(train_pdfs))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model
train_NN_pdf_via_score(model, train_dataloader, optimizer, num_epochs)

# Save the trained model
model_path = "trained_model.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

