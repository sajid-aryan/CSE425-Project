import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class BasicVAE(nn.Module):
    """Basic Variational Autoencoder for feature extraction from music data."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 32):
        super(BasicVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Latent layers
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }


class ConvVAE(nn.Module):
    """Convolutional VAE for spectrograms/MFCC features."""
    
    def __init__(self, input_channels: int = 1, input_height: int = 128, 
                 input_width: int = 128, latent_dim: int = 64):
        super(ConvVAE, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Calculate flattened dimension
        self.encoder_output_dim = 256 * (input_height // 16) * (input_width // 16)
        
        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, self.encoder_output_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 256, self.input_height // 16, self.input_width // 16)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ConvVAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }


class BetaVAE(BasicVAE):
    """Beta-VAE for disentangled representations."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 latent_dim: int = 32, beta: float = 4.0):
        super(BetaVAE, self).__init__(input_dim, hidden_dim, latent_dim)
        self.beta = beta


class ConditionalVAE(nn.Module):
    """Conditional VAE for multi-modal clustering."""
    
    def __init__(self, input_dim: int, condition_dim: int, 
                 hidden_dim: int = 256, latent_dim: int = 32):
        super(ConditionalVAE, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input and condition to latent parameters."""
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode latent and condition to reconstruction."""
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through CVAE."""
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, c)
        
        return {
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }


def vae_loss_function(recon_x: torch.Tensor, x: torch.Tensor, 
                     mu: torch.Tensor, logvar: torch.Tensor, 
                     beta: float = 1.0) -> torch.Tensor:
    """
    VAE loss function combining reconstruction loss and KL divergence.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term
    
    Returns:
        Total loss
    """
    # Reconstruction loss (MSE for continuous data) - use mean instead of sum
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence - average over batch and latent dimensions
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    return recon_loss + beta * kld


def train_vae(model: nn.Module, train_loader, optimizer, device: str, 
              beta: float = 1.0, epochs: int = 100) -> Dict[str, list]:
    """
    Train VAE model.
    
    Args:
        model: VAE model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        beta: Beta parameter for loss
        epochs: Number of training epochs
    
    Returns:
        Dictionary with training losses
    """
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            # Defensive: some feature pipelines can yield NaN/Inf (or extreme values).
            # Sanitizing here prevents NaN loss and training stalls.
            data = torch.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'beta'):  # BetaVAE
                output = model(data)
                loss = vae_loss_function(output['reconstruction'], data, 
                                       output['mu'], output['logvar'], model.beta)
            else:  # BasicVAE or ConvVAE
                output = model(data)
                loss = vae_loss_function(output['reconstruction'], data, 
                                       output['mu'], output['logvar'], beta)
            
            # Skip bad batches
            if not torch.isfinite(loss):
                continue

            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}')
    
    return {'train_losses': train_losses}