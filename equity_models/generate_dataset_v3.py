import numpy as np
import sys
import os

# Add the current directory to path so multiprocessing can find train_v3
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from treys import Card, Deck
from phevaluator.evaluator import evaluate_cards
import random
import time
from multiprocessing import Pool, cpu_count
# Constants migrated from train_v3
STAGE_TO_BOARD_CARDS = {
    "Preflop": 0,
    "Flop": 3,
    "Turn": 4,
    "River": 5
}

TREYS_SUIT_TO_PH = {8: 0, 4: 1, 2: 2, 1: 3}
TREYS_TO_PH = {
    c: (Card.get_rank_int(c) * 4 + TREYS_SUIT_TO_PH[Card.get_suit_int(c)])
    for c in Deck().cards
}

FULL_DECK = Deck().cards

def get_features_v3(hero_hand, board):
    # 1. Card Bits (52 + 52)
    hero_bits = np.zeros(52, dtype=np.float32)
    board_bits = np.zeros(52, dtype=np.float32)
    
    # 2. Histograms (13 ranks + 4 suits)
    rank_counts = np.zeros(13, dtype=np.float32)
    suit_counts = np.zeros(4, dtype=np.float32)
    
    suit_map = {1:0, 2:1, 4:2, 8:3}
    
    all_cards = hero_hand + board
    for c in hero_hand:
        r = Card.get_rank_int(c)
        s = Card.get_suit_int(c)
        hero_bits[r * 4 + suit_map.get(s, 0)] = 1.0
        
    for c in board:
        r = Card.get_rank_int(c)
        s = Card.get_suit_int(c)
        board_bits[r * 4 + suit_map.get(s, 0)] = 1.0
        
    for c in all_cards:
        rank_counts[Card.get_rank_int(c)] += 1.0
        suit_counts[suit_map.get(Card.get_suit_int(c), 0)] += 1.0
        
    # 3. Hand Strength (1)
    if len(board) >= 3:
        try:
            ph_cards = [TREYS_TO_PH[c] for c in hero_hand + board]
            raw_score = evaluate_cards(*ph_cards)
            hand_strength = 1.0 - (raw_score / 7462.0) 
        except Exception:
            hand_strength = 0.5
    else:
        hand_strength = 0.5
        
    # 4. Board Texture (3)
    board_ranks = [Card.get_rank_int(c) for c in board]
    board_suits = [suit_map.get(Card.get_suit_int(c), 0) for c in board]
    
    is_paired = 0.0
    is_trips = 0.0
    if len(board_ranks) > 0:
        counts = [board_ranks.count(r) for r in set(board_ranks)]
        if any(c >= 2 for c in counts): is_paired = 1.0
        if any(c >= 3 for c in counts): is_trips = 1.0
        
    flush_draw = 0.0
    if len(board_suits) > 0:
        s_counts = [board_suits.count(s) for s in set(board_suits)]
        if any(c >= 3 for c in s_counts): flush_draw = 1.0
        
    texture = np.array([is_paired, is_trips, flush_draw], dtype=np.float32)
    
    return np.concatenate([
        hero_bits, 
        board_bits, 
        rank_counts, 
        suit_counts, 
        [hand_strength], 
        texture
    ])

# --- CONFIGURATION ---
STAGE = "Flop"
TOTAL_SAMPLES = 1000000  # 1 Million hands
MC_ITERATIONS = 2500     # High precision
BATCH_SIZE = 1000        # For progress updates


def monte_carlo_equity_v3_int(hero_hand, board, num_samples):
    wins = 0.0
    known_cards = set(hero_hand + board)
    remaining_cards = [c for c in FULL_DECK if c not in known_cards]
    needed_board = 5 - len(board)
    
    h_ph = [TREYS_TO_PH[c] for c in hero_hand]
    b_ph = [TREYS_TO_PH[c] for c in board]
    
    for _ in range(num_samples):
        sampled = random.sample(remaining_cards, 2 + needed_board)
        v_ph = [TREYS_TO_PH[c] for c in sampled[:2]]
        rem_b_ph = [TREYS_TO_PH[c] for c in sampled[2:]]
        full_b_ph = b_ph + rem_b_ph
        
        h_score = evaluate_cards(*(h_ph + full_b_ph))
        v_score = evaluate_cards(*(v_ph + full_b_ph))
        
        if h_score < v_score: wins += 1.0
        elif h_score == v_score: wins += 0.5
    return wins / num_samples

def generate_single_sample(_):
    # We need to recreate the Deck or handle randomness locally in each process
    # To ensure different processes get different hands, we don't seed or we seed with pid+time
    deck = Deck()
    hero_hand = deck.draw(2)
    num_board = STAGE_TO_BOARD_CARDS[STAGE]
    board = deck.draw(num_board) if num_board > 0 else []
    if isinstance(board, int): board = [board]
    
    features = get_features_v3(hero_hand, board)
    equity = monte_carlo_equity_v3_int(hero_hand, board, MC_ITERATIONS)
    return features, equity

def main():
    print(f"--- Starting Large Scale Dataset Generation ---", flush=True)
    print(f"Target: {TOTAL_SAMPLES} samples | MC: {MC_ITERATIONS}", flush=True)
    print(f"Using {cpu_count()} CPU cores...", flush=True)
    
    all_features = []
    all_equities = []
    
    start_time = time.time()
    
    # Process in chunks to manage memory and show progress
    num_chunks = TOTAL_SAMPLES // BATCH_SIZE
    
    with Pool(processes=cpu_count()) as pool:
        for chunk_idx in range(num_chunks):
            # Generate a batch of samples in parallel
            chunk_results = pool.map(generate_single_sample, range(BATCH_SIZE))
            
            for f, e in chunk_results:
                all_features.append(f)
                all_equities.append(e)
            
            if (chunk_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                samples_done = (chunk_idx + 1) * BATCH_SIZE
                percent = (samples_done / TOTAL_SAMPLES) * 100
                rate = samples_done / elapsed
                remaining = (TOTAL_SAMPLES - samples_done) / rate
                print(f"Progress: {percent:5.1f}% | {samples_done}/{TOTAL_SAMPLES} | Rate: {rate:6.1f} samples/s | ETA: {remaining/60:5.1f} min", flush=True)

    # Convert to numpy arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_equities, dtype=np.float32)
    
    print(f"\nGeneration complete! Saving to {OUTPUT_FILE}...", flush=True)
    np.savez_compressed(OUTPUT_FILE, features=X, equities=y)
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time/60:.2f} minutes", flush=True)
    print(f"Final file size: ~{os.path.getsize(OUTPUT_FILE)/1024/1024:.1f} MB", flush=True)

if __name__ == "__main__":
    main()
