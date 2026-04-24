# Poker Equity Model (V5 Universal)

This directory contains the pipeline for generating poker datasets and training a "Universal" Equity Model that predicts hand equity at any stage of the game (Preflop, Flop, Turn, or River).

## Model Overview
The V5 Model is a deep Multi-Layer Perceptron (MLP) trained on ~4 million hands. It replaces stage-specific models with a single neural network capable of generalizing across the entire game state.

## Input Features (125 Dimensions)
The model takes a flat vector of 125 floating-point numbers as input:

1.  **Hero Hand (52 bits)**: A bitmask where `1.0` represents a card held by the Hero.
2.  **Board Cards (52 bits)**: A bitmask where `1.0` represents a card on the board.
3.  **Rank Histogram (13 bits)**: Counts of each rank (2 through Ace) present in the Hero's hand + Board.
4.  **Suit Histogram (4 bits)**: Counts of each suit (Spades, Hearts, Diamonds, Clubs) present in the Hero's hand + Board.
5.  **Hand Strength (1 bit)**: A normalized score calculated via `phevaluator`. `1.0` is the best possible hand (Royal Flush), `0.0` is the worst.
6.  **Board Texture (3 bits)**:
    *   `is_paired`: `1.0` if the board contains a pair.
    *   `is_trips`: `1.0` if the board contains three-of-a-kind.
    *   `flush_draw`: `1.0` if 3 or more cards of the same suit are on the board.

## Directory Structure
*   `training_data/`: Contains `.npz` files (NumPy compressed) for training and testing.
*   `saved_models/`: Contains the trained `.pth` PyTorch models and loss plots.

## How to Use

### 1. Generate Data
To generate test or training datasets for all stages:
```bash
python generate_all_v3.py
```
*Modify `IS_TEST_SET` in the script to toggle between training (1M samples) and testing (5k samples).*

### 2. Train the Model
To train the Universal V5 model on the combined dataset:
```bash
python train_v5.py
```
*Note: This script automatically detects your GPU and moves the entire 4M sample dataset to VRAM for high-speed training.*

### 3. Test the Model
To evaluate the performance of the V5 model across all stages:
```bash
python test_v5.py
```
*This will output MSE, MAE, and a summary table comparing accuracy across Preflop, Flop, Turn, and River.*

## Dependencies
*   `torch` (PyTorch)
*   `numpy`
*   `treys` (Card handling)
*   `phevaluator` (High-speed hand evaluation)
*   `tqdm` (Progress bars)
