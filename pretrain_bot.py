import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from agent import ActorCritic


# =========================
# Dataset (memory efficient)
# =========================
class PokerDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # mmap numpy array
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]).float(),
            torch.tensor(self.y[idx]).long()
        )


# =========================
# Add this to ActorCritic (IMPORTANT)
# =========================
def get_logits(model, state):
    x = torch.relu(model.fc1(state))
    actor_x = torch.relu(model.actor_fc(x))
    logits = model.actor_out(actor_x)
    return logits


# =========================
# Pretraining
# =========================
def pretrain_actor(hero, X, y, device):
    dataset = PokerDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=8192,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # -------- Class imbalance --------
    counts = np.bincount(y)
    weights = counts.sum() / (counts * len(counts))
    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(
        list(hero.fc1.parameters()) +
        list(hero.actor_fc.parameters()) +
        list(hero.actor_out.parameters()),
        lr=1e-3
    )

    hero.train()

    for epoch in range(2):
        total_loss = 0

        # tqdm wrapper
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}", leave=True)

        for xb, yb in progress_bar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = get_logits(hero, xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hero.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # 🔥 update tqdm bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

        avg_loss = total_loss / len(loader)
        print(f"[Pretrain] Epoch {epoch} Avg Loss: {avg_loss:.4f}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset (mmap)...")

    data = np.load("/data/scratch/calvelo/imitation_dataset.npz", mmap_mode="r")
    X = data["X"]
    y = data["y"]

    print("Dataset loaded:")
    print("X:", X.shape)
    print("y:", y.shape)

    hero = ActorCritic(state_dim=119, hidden_dim=256).to(device)

    print("Starting behavior cloning pretraining...")
    pretrain_actor(hero, X, y, device)

    print("Saving pretrained model...")
    torch.save(hero.state_dict(), "hero_pretrained.pth")

    print("Done.")