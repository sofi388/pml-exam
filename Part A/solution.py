import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Define GaussianFourierProjection, ScoreNet, EMA, DDPM here (make sure all previous code is present)
class GaussianFourierProjection(nn.Module):
    def __init__(self, embedding_size=256, scale=30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size // 2) * scale, requires_grad=False)

    def forward(self, t):
        projection = t[:, None] * self.W[None, :]
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class ScoreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ScoreNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, t):
        t_embedding = GaussianFourierProjection()(t)
        x_t = torch.cat([x, t_embedding], dim=-1)
        return self.net(x_t)


class EMA():
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param))

    def apply(self):
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])


class DDPM(nn.Module):
    def __init__(self, network, T=100, beta_1=1e-4, beta_T=2e-2):
        super(DDPM, self).__init__()
        self._network = network
        self.network = lambda x, t: (self._network(x.reshape(-1, 1, 28, 28), (t.squeeze() / T))).reshape(-1, 28 * 28)

        self.T = T
        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        mean = torch.sqrt(alpha_bar_t) * x0
        std = torch.sqrt(1 - alpha_bar_t)
        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        beta_t = self.beta[t].view(-1, 1)
        alpha_t = self.alpha[t].view(-1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        alpha_bar_prev = self.alpha_bar[t - 1].view(-1, 1) if t > 1 else torch.tensor(1.0)

        mean = (torch.sqrt(alpha_bar_prev) * xt - beta_t * epsilon) / torch.sqrt(alpha_bar_t)
        std = torch.sqrt(beta_t)
        return mean + std * epsilon


# Function to experiment with noise prediction vs. x0 prediction
def experiment(xt, t, score_net_noise, score_net_x0):
    epsilon = torch.randn_like(xt)  # random noise to add
    predicted_noise = score_net_noise(xt, t)  # predict noise
    predicted_x0 = score_net_x0(xt, t)  # predict x0

    loss_noise = F.mse_loss(predicted_noise, epsilon)  # Compare predicted noise to the true noise
    loss_x0 = F.mse_loss(predicted_x0, xt)  # Compare predicted x0 to the true xt

    return loss_noise, loss_x0


# Main execution flow
def main():
    # Example data for testing
    xt = torch.randn(64, 28 * 28)  # Example batch of images
    t = torch.randint(0, 100, (64,))  # Example time steps

    # Initialize score networks
    score_net_noise = ScoreNet(input_dim=28 * 28 + 256, hidden_dim=256, output_dim=28 * 28)
    score_net_x0 = ScoreNet(input_dim=28 * 28 + 256, hidden_dim=256, output_dim=28 * 28)

    # Experiment to compare noise prediction vs. x0 prediction
    loss_noise, loss_x0 = experiment(xt, t, score_net_noise, score_net_x0)

    # Print the loss values
    print(f"Loss for Noise Prediction: {loss_noise.item()}")
    print(f"Loss for x0 Prediction: {loss_x0.item()}")


# Run the main function
if __name__ == "__main__":
    main()