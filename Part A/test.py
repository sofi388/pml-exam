import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from solution_ddpm import GaussianFourierProjection, Dense, ScoreNet, ReverseDiffusionPrediction

# Define a function to train the model
def train_diffusion_model(model, dataloader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (x, t) in enumerate(dataloader):
            optimizer.zero_grad()
            # Ensure the data is on the same device as the model
            x, t = x.cuda(), t.cuda()

            # Forward pass
            output = model.predict(x, t, predict_x0=model.predict_x0)  # Either noise or x0 prediction
            # Loss function (MSE between prediction and target)
            target = x if model.predict_x0 else torch.randn_like(x).cuda()
            loss = F.mse_loss(output, target)
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Define a function to evaluate the model
def evaluate_diffusion_model(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i, (x, t) in enumerate(dataloader):
            # Ensure the data is on the same device as the model
            x, t = x.cuda(), t.cuda()
            
            # Predict the target (either noise or x0)
            output = model.predict(x, t, predict_x0=model.predict_x0)
            target = x if model.predict_x0 else torch.randn_like(x).cuda()
            loss = F.mse_loss(output, target)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f'Average MSE Loss: {avg_loss:.4f}')
    return avg_loss

# Visualize the generated samples for comparison
def visualize_comparison(model_1, model_2, dataloader):
    model_1.eval()
    model_2.eval()
    with torch.no_grad():
        for i, (x, t) in enumerate(dataloader):
            x, t = x.cuda(), t.cuda()
            x0_1 = model_1.sample(x.shape)  # Sample using model_1 (predict_x0=False)
            x0_2 = model_2.sample(x.shape)  # Sample using model_2 (predict_x0=True)
            
            # Visualize the first sample
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(x0_1[0, 0].cpu().numpy(), cmap='gray')
            plt.title('Model 1: Predicted Noise')
            plt.subplot(1, 2, 2)
            plt.imshow(x0_2[0, 0].cpu().numpy(), cmap='gray')
            plt.title('Model 2: Predicted x0')
            plt.show()
            break  # Show only the first batch for comparison

# Example usage with MNIST dataset
def main():
    batch_size = 64  # Set a suitable batch size for MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create two models: one for predicting noise (epsilon), one for predicting x0
    model_noise = ReverseDiffusionPrediction(network=ScoreNet(marginal_prob_std=1.0), predict_x0=False).cuda()
    model_x0 = ReverseDiffusionPrediction(network=ScoreNet(marginal_prob_std=1.0), predict_x0=True).cuda()

    # Optimizers
    optimizer_noise = optim.Adam(model_noise.parameters(), lr=1e-4)
    optimizer_x0 = optim.Adam(model_x0.parameters(), lr=1e-4)

    # Training both models
    print("Training model for noise prediction (epsilon)...")
    train_diffusion_model(model_noise, train_loader, optimizer_noise, num_epochs=5)

    print("Training model for x0 prediction...")
    train_diffusion_model(model_x0, train_loader, optimizer_x0, num_epochs=5)

    # Evaluate both models
    print("\nEvaluating model for noise prediction (epsilon)...")
    loss_noise = evaluate_diffusion_model(model_noise, train_loader)

    print("\nEvaluating model for x0 prediction...")
    loss_x0 = evaluate_diffusion_model(model_x0, train_loader)

    # Visualize comparison
    print("\nVisualizing the comparison between the two models...")
    visualize_comparison(model_noise, model_x0, train_loader)

    # Print the comparison of MSE loss
    print(f"\nModel for Noise Prediction (epsilon) - MSE Loss: {loss_noise:.4f}")
    print(f"Model for x0 Prediction - MSE Loss: {loss_x0:.4f}")

# Run the main function
if __name__ == "__main__":
    main()
