import torch
import torch.nn as nn
import numpy as np
import os
import time
from train_v5 import EquityPredictorV5

RANKS = '23456789TJQKA'
SUITS = 'shdc'

def decode_cards(feature_vec):
    hero_bits = feature_vec[:52]
    board_bits = feature_vec[52:104]
    
    def bits_to_cards(bits):
        cards = []
        for i, bit in enumerate(bits):
            if bit > 0.5:
                r = i // 4
                s = i % 4
                cards.append(RANKS[r] + SUITS[s])
        return cards
    
    hero_hand = bits_to_cards(hero_bits)
    board = bits_to_cards(board_bits)
    return f"{hero_hand} | {board}"

# Configuration
STAGES = ["Preflop", "Flop", "Turn", "River"]
MODEL_PATH = os.path.join("saved_models", "equity_model_Universal_v5.pth")
DATA_DIR = "training_data"

def evaluate_stage(model, stage, device):
    npz_path = os.path.join(DATA_DIR, f"poker_data_{stage.lower()}_v3_test.npz")
    
    if not os.path.exists(npz_path):
        print(f"Skipping {stage}: Test data not found.")
        return None
        
    data = np.load(npz_path)
    X_test = torch.from_numpy(data['features']).float().to(device)
    y_test = torch.from_numpy(data['equities']).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        y_pred = model(X_test)
        
    mse = nn.MSELoss()(y_pred, y_test).item()
    mae = torch.mean(torch.abs(y_pred - y_test)).item()
    
    y_test_np = y_test.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()
    errors = np.abs(y_test_np - y_pred_np)
    correlation = np.corrcoef(y_test_np, y_pred_np)[0, 1]
    
    print(f"\n--- {stage} Evaluation ---")
    print(f"MSE: {mse:.6f} | MAE: {mae*100:.2f}% | Corr: {correlation:.4f}")
    
    # 3 Worst Hands for this stage
    worst_idx = np.argsort(errors)[-3:][::-1]
    X_test_np = X_test.cpu().numpy()
    print("Worst 3 Predictions:")
    for idx in worst_idx:
        print(f"  Truth: {y_test_np[idx]:.4f} | Pred: {y_pred_np[idx]:.4f} | Err: {errors[idx]:.4f} | {decode_cards(X_test_np[idx])}")
        
    return {"stage": stage, "mse": mse, "mae": mae, "corr": correlation}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: V5 model not found at {MODEL_PATH}")
        return
        
    model = EquityPredictorV5().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    results = []
    for stage in STAGES:
        res = evaluate_stage(model, stage, device)
        if res:
            results.append(res)
            
    if results:
        print("\n" + "="*60)
        print("V5 UNIVERSAL MODEL SUMMARY")
        print(f"{'Stage':<10} | {'MSE':<10} | {'MAE (%)':<10} | {'Corr':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['stage']:<10} | {r['mse']:.6f} | {r['mae']*100:8.2f}% | {r['corr']:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()
