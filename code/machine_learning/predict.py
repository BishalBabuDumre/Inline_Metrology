import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModal1DCNNVAE(nn.Module):
    def __init__(
        self, 
        in_channels=2304,      # 256 (OES) + 512 (V) + 512 (I) + 1024 (IR)
        seq_len=100,           # 100 timesteps per 1-meter block
        latent_dim=16,         # Compressed fingerprint dimension
        target_dim=2           # Thickness and Porosity
    ):
        super(MultiModal1DCNNVAE, self).__init__()
        
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # ==========================================
        # 1. ENCODER (1D-CNN)
        # ==========================================
        # Downsamples sequence_length: 100 -> 50 -> 25
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(in_channels, 512, kernel_size=5, stride=2, padding=2), # (Batch, 512, 50)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(512, 256, kernel_size=5, stride=2, padding=2),         # (Batch, 256, 25)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1),         # (Batch, 128, 25)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Flatten size: 128 channels * 25 length = 3200
        self.flatten_dim = 128 * 25
        
        # Latent Space Projections (Mean and Log-Variance)
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ==========================================
        # 2. DECODER (1D-Transposed CNN)
        # ==========================================
        self.decoder_fc = nn.Linear(latent_dim, self.flatten_dim)
        
        # Upsamples sequence_length: 25 -> 50 -> 100
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(128, 256, kernel_size=3, stride=1, padding=1), # (Batch, 256, 25)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(256, 512, kernel_size=5, stride=2, padding=2, output_padding=1), # (Batch, 512, 50)
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(512, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1) # (Batch, 2304, 100)
        )

        # ==========================================
        # 3. REGRESSION HEAD (Thickness & Porosity)
        # ==========================================
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, target_dim) # Output: [Thickness, Porosity]
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = self.encoder_conv(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def decode(self, z):
        h_flat = self.decoder_fc(z)
        h = h_flat.view(h_flat.size(0), 128, 25)
        x_recon = self.decoder_conv(h)
        return x_recon

    def forward(self, x):
        # 1. Encode
        mu, logvar = self.encode(x)
        
        # 2. Sample latent space
        z = self.reparameterize(mu, logvar) if self.training else mu
        
        # 3. Reconstruct multi-modal inputs
        x_recon = self.decode(z)
        
        # 4. Predict targets from latent fingerprint z
        targets_pred = self.predictor(z)
        
        return x_recon, targets_pred, mu, logvar


# =====================================================================
# LOSS FUNCTION & TRAINING LOOP EXAMPLE
# =====================================================================

def vae_loss_function(x_recon, x, targets_pred, targets_true, mu, logvar, beta=0.001, gamma=1.0):
    """
    Combined Loss: Reconstruction + KL-Divergence + Regression MSE
    """
    # 1. Reconstruction Loss (MSE over all 2304 channels and 100 timesteps)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # 2. KL Divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    
    # 3. Supervised Prediction Loss (Thickness & Porosity)
    pred_loss = F.mse_loss(targets_pred, targets_true, reduction='mean')
    
    total_loss = recon_loss + (beta * kl_loss) + (gamma * pred_loss)
    
    return total_loss, recon_loss, kl_loss, pred_loss


class MultiModalOESDataset(Dataset):
    def __init__(self, csv_filepath, targets_dict=None):
        """
        Parameters:
            csv_filepath (str): Path to the CSV file.
            targets_dict (dict, optional): Dictionary mapping sample_id -> [thickness, porosity].
                                           Example: {0: [0.85, 0.12], 1: [0.82, 0.15], ...}
        """
        # Load CSV (header=0, column 0 is index)
        df = pd.read_csv(csv_filepath, index_col=0)
        
        # Extract metadata and feature columns
        self.sample_ids = df['sample_id'].unique()
        feature_cols = df.columns[2:]  # Columns 3 onward (index 2:): 2304 feature columns
        
        # 1. Channel-wise Standardization across the entire dataset
        # Fits StandardScaler over all 2,000 rows for 2,304 features
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        
        self.samples = []
        self.targets = []
        
        # 2. Group by sample_id to construct (2304, 100) tensors
        for s_id in self.sample_ids:
            sample_df = df[df['sample_id'] == s_id].sort_values(by='measurement_id')
            
            # Extract 2304 feature columns -> Shape: (100, 2304)
            features = sample_df[feature_cols].values
            
            # Transpose to PyTorch Conv1D format: (Channels, Sequence_Length) -> (2304, 100)
            tensor_x = torch.tensor(features, dtype=torch.float32).T
            self.samples.append(tensor_x)
            
            # Attach targets [Thickness, Porosity] if provided, else use placeholders
            if targets_dict and s_id in targets_dict:
                self.targets.append(torch.tensor(targets_dict[s_id], dtype=torch.float32))
            else:
                self.targets.append(torch.tensor([0.0, 0.0], dtype=torch.float32))
                
        # Stack all samples into a unified batch tensor: (N_samples, 2304, 100)
        self.samples = torch.stack(self.samples)
        self.targets = torch.stack(self.targets)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


# =====================================================================
# REAL DATA LOADING AND TRAINING EXECUTION
# =====================================================================

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    csv_file_path = PROJECT_ROOT / 'data' / 'measurement' / 'Measurement_Data.csv'
    truth_path = PROJECT_ROOT / 'data' / 'measurement' / 'Sample_Info.csv'

    df_truth = pd.read_csv(truth_path, index_col=0)
    # Ground-truth targets dict mapping each sample_id (0 to 19) to [Thickness, Porosity]
    sample_targets = {
    k: [t, p] 
    for k, t, p in zip(df_truth['sample_id'], df_truth['thickness'], df_truth['porosity'])
    }
   
    # Initialize PyTorch Dataset & DataLoader
    dataset = MultiModalOESDataset(csv_filepath=csv_file_path, targets_dict=sample_targets)
    
    # DataLoader handles batching, shuffling, and iterating over the 20 samples
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Model Setup
    in_channels = 2304
    seq_len = 100
    latent_dim = 16
    
    model = MultiModal1DCNNVAE(in_channels=in_channels, seq_len=seq_len, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training Loop over Real Data
    model.train()
    print(f"Dataset successfully created!")
    print(f"Total no. of samples: {len(dataset)}")
    print(f"Single sample shape:   {dataset[0][0].shape}  -> (Channels, Sequence_Length)\n")
    
    for epoch in range(50):  # Example: 5 Epochs
        total_epoch_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            # batch_x shape: (batch_size, 2304, 100)
            # batch_y shape: (batch_size, 2)
            
            optimizer.zero_grad()
            x_recon, y_pred, mu, logvar = model(batch_x)
            
            loss, recon, kl, pred = vae_loss_function(
                x_recon, batch_x, y_pred, batch_y, mu, logvar
            )
            
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()
            
        print(f"Epoch {epoch}/50 - Total Loss: {total_epoch_loss / len(train_loader):.4f}")
