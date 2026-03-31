import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import csv

from agent import ActorCritic
from vectorized_env import VectorizedPokerEnv

training_phase = 0


number_of_villains = 5

villain_agent = "hero_agent_" + str(training_phase) + ".pth"
next_agent = "hero_agent_" + str(training_phase + 1) + ".pth"

def train(num_envs=32, target_episodes=200000, lr=5e-5, gamma=0.99):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    envs = VectorizedPokerEnv(num_envs=num_envs)
    
    hero = ActorCritic(state_dim=119, hidden_dim=256).to(device)
    villains = [ActorCritic(state_dim=119, hidden_dim=256).to(device) for _ in range(number_of_villains)]
    
    if os.path.exists(villain_agent):
        print(f"Warm-starting Hero with {villain_agent}...")
        try:
            weights = torch.load(villain_agent, weights_only=True)
            if 'fc1.weight' in weights and weights['fc1.weight'].shape[1] == 118:
                w = weights['fc1.weight']
                padded_w = torch.cat([w, torch.zeros(w.shape[0], 1).to(w.device)], dim=1)
                weights['fc1.weight'] = padded_w
            hero.load_state_dict(weights)
            print("Successfully loaded Hero.")
        except Exception as e:
            print(f"Failed to load Hero from {villain_agent}:", e)

    print("Assigning historical models to Villains...")
    available_history = [f"hero_agent_{i}.pth" for i in range(training_phase + 1)]
    existing_history = [m for m in available_history if os.path.exists(m)]
    
    for i, v in enumerate(villains):
        if existing_history:
            chosen_model = random.choice(existing_history)
            try:
                v_weights = torch.load(chosen_model, weights_only=True)
                if 'fc1.weight' in v_weights and v_weights['fc1.weight'].shape[1] == 118:
                    w = v_weights['fc1.weight']
                    padded_w = torch.cat([w, torch.zeros(w.shape[0], 1).to(w.device)], dim=1)
                    v_weights['fc1.weight'] = padded_w
                v.load_state_dict(v_weights)
                print(f"Seat {i+1} (Villain) is playing as: {chosen_model}")
            except Exception as e:
                print(f"Failed to load {chosen_model} for Seat {i+1}:", e)
        v.eval()
    
    optimizer = optim.Adam(hero.parameters(), lr=lr)
    
    print(f"Starting Training (1 Hero vs {number_of_villains} Villains) over {num_envs} Parallel Environments...")
    
    cumulative_hero_reward = 0.0
    cumulative_villain_reward = 0.0
    episodes_completed = 0
    last_computed_loss = 0.0
    
    batch_size = 1000
    batch_states = []
    batch_actions = []
    batch_valid_actions = []
    batch_returns = []
    
    env_trajectories = [[] for _ in range(num_envs)]
    
    history_episodes = []
    history_hero_loss = []
    history_hero_reward = []
    history_villain_reward = []
    
    results = envs.reset()
    
    while episodes_completed < target_episodes:
        env_actions = [0] * num_envs
        # Group environments by current_player
        player_to_envs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        
        for i, (current_player, state, valid_actions) in enumerate(results):
            player_to_envs[current_player].append((i, state, valid_actions))
            
        for player, group in player_to_envs.items():
            if len(group) == 0:
                continue
            
            indices = [item[0] for item in group]
            states = [item[1] for item in group]
            valid_action_lists = [item[2] for item in group]

            states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            
            if player == 0:
                with torch.no_grad():
                    probs, _ = hero.forward(states_tensor)
                    mask = torch.full(probs.shape, -float('inf')).to(device)
                    for batch_idx, valids in enumerate(valid_action_lists):
                        for v in valids:
                            mask[batch_idx, v] = 0
                            
                    logits = torch.log(probs + 1e-9) + mask
                    masked_probs = F.softmax(logits, dim=-1)
                    
                    m = torch.distributions.Categorical(masked_probs)
                    actions = m.sample()
                    
                for batch_idx, env_idx in enumerate(indices):
                    env_actions[env_idx] = actions[batch_idx].item()
                    env_trajectories[env_idx].append((states[batch_idx], valid_action_lists[batch_idx], actions[batch_idx].item()))
            else:
                with torch.no_grad():
                    probs, _ = villains[player - 1].forward(states_tensor)
                    mask = torch.full(probs.shape, -float('inf')).to(device)
                    for batch_idx, valids in enumerate(valid_action_lists):
                        for v in valids:
                            mask[batch_idx, v] = 0
                    logits = torch.log(probs + 1e-9) + mask
                    masked_probs = F.softmax(logits, dim=-1)
                    
                    m = torch.distributions.Categorical(masked_probs)
                    actions = m.sample()
                    
                    for batch_idx, env_idx in enumerate(indices):
                        env_actions[env_idx] = actions[batch_idx].item()

        step_results = envs.step(env_actions)
        results = []
        
        for i, res in enumerate(step_results):
            current_player, state, valid_actions, rewards, done = res
            results.append((current_player, state, valid_actions))
            
            if done:
                R = rewards[0]
                hero_reward = R
                villains_reward = sum(rewards[1:]) / number_of_villains
                
                cumulative_hero_reward += hero_reward
                cumulative_villain_reward += villains_reward
                episodes_completed += 1
                
                episode_returns = []
                for _ in range(len(env_trajectories[i])):
                    episode_returns.insert(0, R)
                    R = R * gamma
                    
                for step_idx, (s, valids, a) in enumerate(env_trajectories[i]):
                    batch_states.append(s)
                    batch_valid_actions.append(valids)
                    batch_actions.append(a)
                    batch_returns.append(episode_returns[step_idx])
                    
                env_trajectories[i] = []
                
                if episodes_completed % 500 == 0:
                    avg_hero = cumulative_hero_reward / 500.0
                    avg_villains = cumulative_villain_reward / 500.0
                    print(f"Episode {episodes_completed}: Hero Loss: {last_computed_loss:.4f} | Avg Rewards (Last 500): Hero {avg_hero:.4f}, Villains {avg_villains:.4f}")
                    
                    history_episodes.append(episodes_completed)
                    history_hero_loss.append(last_computed_loss)
                    history_hero_reward.append(avg_hero)
                    history_villain_reward.append(avg_villains)
                    
                    cumulative_hero_reward = 0.0
                    cumulative_villain_reward = 0.0

        if len(batch_states) >= batch_size:
            entropy_weight = 0.05
            
            states_tensor = torch.tensor(np.array(batch_states), dtype=torch.float32).to(device)
            actions_tensor = torch.tensor(batch_actions, dtype=torch.int64).to(device)
            returns_tensor = torch.tensor(batch_returns, dtype=torch.float32).to(device)
            
            if len(returns_tensor) > 1:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
                
            probs, values = hero.forward(states_tensor)
            values_tensor = values.squeeze(-1)
            
            mask = torch.full(probs.shape, -float('inf')).to(device)
            for batch_idx, valids in enumerate(batch_valid_actions):
                for v in valids:
                    mask[batch_idx, v] = 0
                    
            logits = torch.log(probs + 1e-9) + mask
            masked_probs = F.softmax(logits, dim=-1)
            
            m = torch.distributions.Categorical(masked_probs)
            log_probs = m.log_prob(actions_tensor)
            entropies = m.entropy()
            
            advantages = returns_tensor - values_tensor.detach()
            
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values_tensor, returns_tensor)
            entropy_bonus = entropies.mean()
            
            loss = actor_loss + 0.5 * critic_loss - (entropy_weight * entropy_bonus)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hero.parameters(), max_norm=1.0)
            optimizer.step()
            
            last_computed_loss = loss.item()
            
            batch_states = []
            batch_actions = []
            batch_valid_actions = []
            batch_returns = []

    envs.close()

    print("Saving Hero model to " + next_agent + "...")
    torch.save(hero.state_dict(), next_agent)
    
    csv_filename = 'training_metrics' + str(training_phase + 1) + '.csv'
    print("Saving training metrics to " + csv_filename + "...")
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Hero_Loss', 'Hero_Avg_Reward', 'Villains_Avg_Reward'])
        for ep, loss, h_rew, v_rew in zip(history_episodes, history_hero_loss, history_hero_reward, history_villain_reward):
            writer.writerow([ep, loss, h_rew, v_rew])

    print("Plotting and saving training_metrics.png...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_episodes, history_hero_loss, label='Hero Loss', color='blue')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.title('Hero Training Loss Over Time')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_episodes, history_hero_reward, label='Hero Avg Reward', color='green')
    plt.plot(history_episodes, history_villain_reward, label='Villains Avg Reward', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward (Last 500)')
    plt.title('Average Reward Over Time')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics' + str(training_phase + 1) + '.png')
    plt.close()

if __name__ == "__main__":
    train()
