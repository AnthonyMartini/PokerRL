import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

# Training constants
STAGES = ["Preflop", "Flop", "Turn", "River"]
DATA_DIR = "training_data"
MODEL_SAVE_DIR = "saved_models"
BATCH_SIZE = 16384  # Large batch for GPU efficiency
EPOCHS = 100
LEARNING_RATE = 0.001
VAL_SPLIT = 0.05

class UniversalPokerDataset(Dataset):
    def __init__(self, data_dir, stages, device):
        all_features = []
        all_equities = []
        
        for stage in stages:
            path = os.path.join(data_dir, f"poker_data_{stage.lower()}_v3.npz")
            if not os.path.exists(path):
                print(f"Warning: Dataset for {stage} not found at {path}")
                continue
                
            print(f"Loading {stage} data...")
            data = np.load(path)
            all_features.append(data['features'])
            all_equities.append(data['equities'])
            
        print("Concatenating datasets...")
        self.features = torch.from_numpy(np.concatenate(all_features, axis=0)).float().to(device)
        self.equities = torch.from_numpy(np.concatenate(all_equities, axis=0)).float().unsqueeze(1).to(device)
        
        print(f"Universal Dataset Loaded. Total samples: {len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.equities[idx]

class EquityPredictorV5(nn.Module):
    """
    V5: Same core architecture as V4, but trained on the universal dataset.
    """
    def __init__(self, input_dim=125):
        super(EquityPredictorV5, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable TF32 for Ampere cards
    torch.set_float32_matmul_precision('high')

    full_dataset = UniversalPokerDataset(DATA_DIR, STAGES, device)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = EquityPredictorV5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_SAVE_DIR, 'equity_model_Universal_v5.pth')
    
    print(f"Starting training of V5 Universal Equity Model...")
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for inputs, labels in pbar:
            # Data is already on device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(epoch_val_loss)
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  --> Best model saved with Val Loss: {best_val_loss:.6f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('V5 Universal Equity Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(MODEL_SAVE_DIR, 'loss_plot_Universal_v5.png'))
    plt.close()
    
    print(f"Training complete. Model saved to {save_path}")

if __name__ == '__main__':
    train()
