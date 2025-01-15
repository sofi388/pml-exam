import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define neural network architecture (based on the model used in the MNIST DDPM solution)
class DiffusionModel(nn.Module):
    def __init__(self, model_type='epsilon'):
        super(DiffusionModel, self).__init__()
        self.model_type = model_type
        
        # Example architecture: CNN with several layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, 784)  # for x0 output

    def forward(self, x, t):
        # Apply CNN layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        
        if self.model_type == 'epsilon':
            # Predict epsilon (noise)
            return self.fc2(x).view(-1, 28, 28)
        
        elif self.model_type == 'mu':
            # Predict mu
            return self.fc2(x).view(-1, 28, 28)  # Output as a mean

        elif self.model_type == 'x0':
            # Predict x0
            return self.fc2(x).view(-1, 28, 28)  # Output as the denoised image

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the model (example for one model type)
def train_model(model, train_loader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for data, _ in train_loader:
            optimizer.zero_grad()
            
            # Apply diffusion model training (e.g., predicting epsilon, mu, or x0)
            predicted = model(data, t=0)  # t=0 for simplicity, can be expanded for timesteps
            loss = criterion(predicted, data)  # Here we are using MSE
            
            loss.backward()
            optimizer.step()

# Initialize models
epsilon_model = DiffusionModel(model_type='epsilon')
mu_model = DiffusionModel(model_type='mu')
x0_model = DiffusionModel(model_type='x0')

# Define optimizer and loss
optimizer = optim.Adam(list(epsilon_model.parameters()) + list(mu_model.parameters()) + list(x0_model.parameters()), lr=1e-4)
criterion = nn.MSELoss()

# Train models
train_model(epsilon_model, train_loader, optimizer, criterion)
train_model(mu_model, train_loader, optimizer, criterion)
train_model(x0_model, train_loader, optimizer, criterion)

# Evaluate and compare the models (MSE, PSNR, SSIM, etc.)
