"""
Sample script to run a single poker game in a 5-player environment.
Hero vs 4 Villains.
"""

import torch
import os
from treys import Card

from poker_env import PokerEnv
from agent import PolicyNetwork, ActorCritic

hero_model = "hero_agent_1.pth"
villain_model = "hero_agent_1.pth"


def print_cards(cards, prefix="Cards:"):
    if type(cards) is int:
        cards = [cards]
    if len(cards) == 0:
        print(f"{prefix} None")
        return
        
    pretty_cards = [Card.int_to_pretty_str(c) for c in cards]
    print(f"{prefix} {' '.join(pretty_cards)}")


def test_game():
    env = PokerEnv(num_players=6)
    
    hero = ActorCritic(state_dim=119, hidden_dim=256)
    villains = [ActorCritic(state_dim=119, hidden_dim=256) for _ in range(5)]
    
    if os.path.exists(hero_model):
        weights = torch.load(hero_model, weights_only=True)
        if 'fc1.weight' in weights and weights['fc1.weight'].shape[1] == 118:
            w = weights['fc1.weight']
            padded_w = torch.cat([w, torch.zeros(w.shape[0], 1).to(w.device)], dim=1)
            weights['fc1.weight'] = padded_w
        hero.load_state_dict(weights)
        print("Loaded saved weights for Hero.")
        
    if os.path.exists(villain_model):
        print("Loading pre-trained " + villain_model + " into Villains...")
        try:
            weights = torch.load(villain_model, weights_only=True)
            if 'fc1.weight' in weights and weights['fc1.weight'].shape[1] == 118:
                w = weights['fc1.weight']
                padded_w = torch.cat([w, torch.zeros(w.shape[0], 1).to(w.device)], dim=1)
                weights['fc1.weight'] = padded_w
            for v in villains:
                v.load_state_dict(weights)
        except Exception as e:
            print("Failed to load " + villain_model + ":", e)
            
    hero.eval()
    for v in villains:
        v.eval()
    
    agents = [hero] + villains
    
    print("=========================================")
    print("Starting Sample 6-Player Poker Game")
    print("=========================================")
    
    try:
        current_player, state = env.reset()
    except Exception as e:
        print("Failed to initialize game:", e)
        return
        
    for i in range(6):
        name = "Hero" if i == 0 else f"Villain {i}"
        print(f"{name} Stack: ${env.stacks[i]}")
        
    print(f"Pot starts with Blinds = ${env.pot}")
    print("\n--- Pre-flop ---")
    
    for i in range(6):
        name = "Hero" if i == 0 else f"Villain {i}"
        print_cards(env.hands[i], f"{name} Hand:")
        
    print("")

    stage_names = ["Pre-flop", "Flop", "Turn", "River", "Showdown"]
    current_stage = 0
    
    done = False
    
    while not done:
        # Check if we moved to a new betting round
        if env.stage > current_stage:
            current_stage = env.stage
            if current_stage <= 4:
                print(f"\n--- {stage_names[current_stage]} ---")
                print_cards(env.board, "Community Board:")
                print(f"Current Pot: ${env.pot}")
                for i in range(6):
                    if env.active[i]:
                        name = "Hero" if i == 0 else f"Villain {i}"
                        print(f"{name} Stack: ${env.stacks[i]}")
                print("")

        valid_actions = env.get_valid_actions(current_player)
        action_names = {0: "Fold", 1: "Call/Check", 2: "Raise"}
        
        with torch.no_grad():
            if current_player == 0:
                action, _, _, _ = agents[current_player].select_action(state, valid_actions)
            else:
                action, _, _, _ = agents[current_player].select_action(state, valid_actions)
                
        name = "Hero" if current_player == 0 else f"Villain {current_player}"
        print(f"{name} chooses: {action_names[action]}")
            
        try:
            current_player, next_state, rewards, done = env.step(action)
        except Exception as e:
            print("Error during step:", e)
            done = True
            
        state = next_state
        
    print("\n=========================================")
    if len(env.winners) > 0:
        winner_names = ["Hero" if w == 0 else f"Villain {w}" for w in env.winners]
        print(f"Game Over! Winners: {', '.join(winner_names)}")
    else:
        print("Game Over! Tied / Logic Fallback.")
        
    print("\nFinal Rewards:")
    for i in range(6):
        name = "Hero" if i == 0 else f"Villain {i}"
        print(f"{name}: {rewards[i] * env.starting_stack * env.num_players:.2f}")
    
    if len(env.board) >= 5 and len(env.winners) > 0:
        print("\nFinal Setup:")
        for i in range(6):
            if env.active[i]:
                name = "Hero" if i == 0 else f"Villain {i}"
                print_cards(env.hands[i], f"{name} Hand:")
                
        print_cards(env.board, "Community Board:")
        for i in range(6):
            if env.active[i]:
                try:
                    score = env.evaluator.evaluate(env.board, env.hands[i])
                    rank_class = env.evaluator.get_rank_class(score)
                    name = "Hero" if i == 0 else f"Villain {i}"
                    print(f"{name} Hand Rank: {env.evaluator.class_to_string(rank_class)}")
                except:
                    pass

if __name__ == "__main__":
    test_game()
