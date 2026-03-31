"""
Script to run 1000 poker games in a 5-player environment and average the gains.
Hero vs 4 Villains.
"""

import torch
import os
import numpy as np

from poker_env import PokerEnv
from agent import PolicyNetwork, ActorCritic

hero_model = "hero_agent_9.pth"
villain_model = "hero_agent_8.pth"

def test_average(num_games=1000):
    env = PokerEnv(num_players=2)
    
    hero = ActorCritic(state_dim=119, hidden_dim=256)
    villains = [ActorCritic(state_dim=119, hidden_dim=256) for _ in range(1)]
    
    if os.path.exists(hero_model):
        weights = torch.load(hero_model, weights_only=True)
        if 'fc1.weight' in weights and weights['fc1.weight'].shape[1] == 118:
            w = weights['fc1.weight']
            padded_w = torch.cat([w, torch.zeros(w.shape[0], 1).to(w.device)], dim=1)
            weights['fc1.weight'] = padded_w
        hero.load_state_dict(weights)
        print(f"Loaded saved weights for Hero ({hero_model}).")
    else:
        print(f"Hero model {hero_model} not found.")
        
    if os.path.exists(villain_model):
        print(f"Loading pre-trained {villain_model} into Villains...")
        try:
            weights = torch.load(villain_model, weights_only=True)
            if 'fc1.weight' in weights and weights['fc1.weight'].shape[1] == 118:
                w = weights['fc1.weight']
                padded_w = torch.cat([w, torch.zeros(w.shape[0], 1).to(w.device)], dim=1)
                weights['fc1.weight'] = padded_w
            for v in villains:
                v.load_state_dict(weights)
        except Exception as e:
            print(f"Failed to load {villain_model}:", e)
    else:
        print(f"Villain model {villain_model} not found.")
            
    hero.eval()
    for v in villains:
        v.eval()
    
    agents = [hero] + villains
    
    print(f"=========================================")
    print(f"Starting {num_games} Sample 6-Player Poker Games")
    print(f"=========================================")
    
    total_hero_reward = 0.0
    total_villain_reward = 0.0
    
    hero_wins = 0
    villain_wins = 0
    ties = 0

    for game in range(1, num_games + 1):
        try:
            current_player, state = env.reset()
        except Exception as e:
            print("Failed to initialize game:", e)
            continue
            
        done = False
        
        while not done:
            valid_actions = env.get_valid_actions(current_player)
            
            with torch.no_grad():
                if current_player == 0:
                    action, _, _, _ = agents[current_player].select_action(state, valid_actions)
                else:
                    action, _, _, _ = agents[current_player].select_action(state, valid_actions)
                    
            try:
                current_player, next_state, rewards, done = env.step(action)
            except Exception as e:
                done = True
                rewards = [0] * 5
                
            state = next_state
            
        # Add to total rewards
        for i in range(6):
            absolute_reward = rewards[i] * env.starting_stack * env.num_players
            if i == 0:
                total_hero_reward += absolute_reward
            else:
                total_villain_reward += absolute_reward
                
        if len(env.winners) == 1:
            if env.winners[0] == 0:
                hero_wins += 1
            else:
                villain_wins += 1
        elif len(env.winners) > 1:
            if 0 in env.winners:
                hero_wins += 1 / len(env.winners)
                villain_wins += (len(env.winners) - 1) / len(env.winners)
            else:
                villain_wins += 1
        else:
            ties += 1

        if game % 100 == 0:
            print(f"Completed {game}/{num_games} games...")
            
    avg_hero_reward = total_hero_reward / num_games
    avg_villain_reward = total_villain_reward / (num_games * 5) # average per villain per game
    sum_villains_avg_per_game = total_villain_reward / num_games
    
    print("\n=========================================")
    print(f"Results after {num_games} games:")
    print("=========================================")
    print(f"Hero Wins: {hero_wins:.2f}")
    print(f"Villain Wins (total for all 5): {villain_wins:.2f}")
    print(f"Ties: {ties}")
    print(f"-----------------------------------------")
    print(f"Total Hero Reward: ${total_hero_reward:.2f}")
    print(f"Total Villains Reward: ${total_villain_reward:.2f}")
    print(f"-----------------------------------------")
    print(f"Average Hero Reward per game: ${avg_hero_reward:.2f}")
    print(f"Average total Villains Reward per game: ${sum_villains_avg_per_game:.2f}")
    print(f"Average Reward per Villain per game: ${avg_villain_reward:.2f}")

if __name__ == "__main__":
    test_average()
