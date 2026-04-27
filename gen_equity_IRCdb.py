import numpy as np
from collections import defaultdict
import re
import os
from phevaluator import evaluate_cards
import random

# ========= CONFIG =========
OUTPUT_FILE = "/data/scratch/calvelo/imitation_good_dataset.npz"

RANKS = "23456789TJQKA"
SUITS = "shdc"

# ========= CARD HELPERS =========

def parse_card(card_str):
    r = RANKS.index(card_str[0])
    s = SUITS.index(card_str[1])
    return r * 4 + s

def cards_to_vec(cards):
    vec = np.zeros(52, dtype=np.float32)
    for c in cards:
        vec[parse_card(c)] = 1.0
    return vec

# ========= ACTION MAPPING =========

def map_action(a):
    if a == '-' or a == '':
        return None
    if 'f' in a:
        return 0
    if 'c' in a or 'k' in a:
        return 1
    if 'b' in a or 'r' in a:
        return 2
    return None


def compute_equity(hand, board, n_players=2, num_samples=100):
    wins = 0

    known_cards = set(hand + board)
    deck = [r+s for r in "23456789TJQKA" for s in "shdc"]
    deck = [c for c in deck if c not in known_cards]

    for _ in range(num_samples):
        random.shuffle(deck)

        opp_hand = deck[:2]
        remaining = deck[2:]

        full_board = board.copy()
        needed = 5 - len(full_board)
        full_board += remaining[:needed]

        hero_score = evaluate_cards(*(hand + full_board))
        opp_score = evaluate_cards(*(opp_hand + full_board))

        if hero_score < opp_score:  # lower is better
            wins += 1

    return wins / num_samples

# ========= LOAD FILES =========

def load_hdb(path):
    boards = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            hand_id = parts[0].strip()

            # extract cards robustly
            cards = [t for t in parts if re.match(r"[2-9TJQKA][shdc]", t)]

            if len(cards) >= 3:  # at least flop
                boards[hand_id] = cards

    print(f"Loaded boards: {len(boards)}")
    return boards


def load_hroster(path):
    roster = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            hand_id = parts[0].strip()
            players = parts[2:]
            roster[hand_id] = players

    print(f"Loaded rosters: {len(roster)}")
    return roster


def load_pdb(path):
    hands = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            hand_id = parts[1].strip()
            hands[hand_id].append(parts)

    return hands

# ========= STATE BUILDER =========

def build_state(hand_vec, board_vec, stage, position, n_players, equity):

    stage_vec = np.zeros(4, dtype=np.float32)
    stage_vec[stage] = 1.0

    pos_vec = np.zeros(10, dtype=np.float32)
    position = min(position, 9)
    pos_vec[position] = 1.0

    n_players_norm = np.array([n_players / 10.0], dtype=np.float32)

    equity_vec = np.array([equity], dtype=np.float32)

    return np.concatenate([
        hand_vec,
        board_vec,
        stage_vec,
        pos_vec,
        n_players_norm,
        equity_vec
    ])

# ========= DATASET BUILDER =========

def build_dataset(pdb_path, hdb_path, hroster_path):

    boards = load_hdb(hdb_path)
    rosters = load_hroster(hroster_path)
    hands = load_pdb(pdb_path)
    equity_cache = {}

    X, y = [], []

    matched = 0
    total = 0

    for hand_id, rows in hands.items():
        total += 1

        if hand_id not in boards or hand_id not in rosters:
            continue

        matched += 1

        board = boards[hand_id]
        n_players = len(rosters[hand_id])

        for row in rows:

            try:
                seat = int(row[3]) - 1
                actions = row[4:8]
            except:
                continue

            # detect cards
            cards = [t for t in row if re.match(r"[2-9TJQKA][shdc]", t)]
            cards = cards[-2:] if len(cards) >= 2 else []

            hand_vec = cards_to_vec(cards) if len(cards) == 2 else np.zeros(52)

            if len(cards) != 2:
                continue

            for stage in range(4):

                if stage == 0:
                    continue  # skip preflop

                if stage >= len(actions):
                    continue

                action = map_action(actions[stage])
                if action is None:
                    continue

                if stage == 1:
                    board_cards = board[:3]
                elif stage == 2:
                    board_cards = board[:4]
                else:
                    board_cards = board[:5]

                board_vec = cards_to_vec(board_cards)

                key = (
                    tuple(sorted(cards)),
                    tuple(sorted(board_cards)),
                    n_players
                )

                if key in equity_cache:
                    equity = equity_cache[key]
                else:
                    equity = compute_equity(cards, board_cards, n_players, num_samples=50)
                    equity_cache[key] = equity

                # better filtering
                if equity < 0.1 and action != 0:
                    continue
                elif equity > 0.8 and action != 2:
                    continue

                state = build_state(
                    hand_vec,
                    board_vec,
                    stage,
                    seat,
                    n_players,
                    equity
                )

                X.append(state)
                y.append(action)

    print(f"Matched hands: {matched}/{total}")
    print(f"Samples generated: {len(X)}")

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

# ========= RUN =========
if __name__ == "__main__":

    ROOT_DIR = "/home/calvelo/temp/IRCdata"

    all_X, all_y = [], []

    for outer in os.listdir(ROOT_DIR):

        outer_path = os.path.join(ROOT_DIR, outer)
        if not os.path.isdir(outer_path):
            continue

        # go into h1-nobots
        mid_path = os.path.join(outer_path, "h1-nobots")
        if not os.path.isdir(mid_path):
            continue

        # go into actual month folder (e.g. 199804)
        for inner in os.listdir(mid_path):

            inner_path = os.path.join(mid_path, inner)
            if not os.path.isdir(inner_path):
                continue

            hdb_path = os.path.join(inner_path, "hdb")
            hroster_path = os.path.join(inner_path, "hroster")
            pdb_dir = os.path.join(inner_path, "pdb")

            if not os.path.exists(hdb_path):
                print(f"Skipping {outer}/{inner}: missing hdb")
                continue

            if not os.path.exists(hroster_path):
                print(f"Skipping {outer}/{inner}: missing hroster")
                continue

            if not os.path.isdir(pdb_dir):
                print(f"Skipping {outer}/{inner}: missing pdb folder")
                continue

            pdb_files = [f for f in os.listdir(pdb_dir) if f.startswith("pdb.")]
            if len(pdb_files) == 0:
                print(f"Skipping {outer}/{inner}: no pdb files")
                continue

            print(f"\n=== Processing {outer}/{inner} ===")

            for pdb_file in pdb_files:

                pdb_path = os.path.join(pdb_dir, pdb_file)

                try:
                    X, y = build_dataset(pdb_path, hdb_path, hroster_path)

                    if len(X) > 0:
                        all_X.append(X)
                        all_y.append(y)

                except Exception as e:
                    print(f"Skipping {pdb_file}: {e}")

    if not all_X:
        print("No data found. (Check parsing assumptions)")
        exit()

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)

    print(f"\nFINAL DATASET SIZE: {len(X)}")

    np.savez(OUTPUT_FILE, X=X, y=y)
    print(f"Saved → {OUTPUT_FILE}")