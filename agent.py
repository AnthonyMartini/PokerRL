import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=108, action_dim=3, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state, valid_actions=None):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        if valid_actions is not None:
            # Create a mask of negative infinity
            mask = torch.full(logits.shape, -float('inf')).to(logits.device)
            # Set valid action indices to 0 so they don't change the logit value
            mask[0, valid_actions] = 0 
            logits = logits + mask
            
        return F.softmax(logits, dim=-1)

    def select_action(self, state, valid_actions):
        device = next(self.parameters()).device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # Pass valid_actions to forward pass
        probs = self.forward(state, valid_actions)
        
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), m.entropy()

class ActorCritic(nn.Module):
    def __init__(self, state_dim=119, action_dim=3, hidden_dim=256):
        super(ActorCritic, self).__init__()
        # Shared features
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Actor head
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, valid_actions=None):
        x = F.relu(self.fc1(state))
        
        # Actor
        actor_x = F.relu(self.actor_fc(x))
        logits = self.actor_out(actor_x)
        
        if valid_actions is not None:
            mask = torch.full(logits.shape, -float('inf')).to(logits.device)
            mask[0, valid_actions] = 0 
            logits = logits + mask
            
        probs = F.softmax(logits, dim=-1)
        
        # Critic
        critic_x = F.relu(self.critic_fc(x))
        value = self.critic_out(critic_x)
        
        return probs, value
        
    def select_action(self, state, valid_actions):
        device = next(self.parameters()).device
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs, value = self.forward(state, valid_actions)
        
        m = Categorical(probs)
        action = m.sample()
        
        return action.item(), m.log_prob(action), m.entropy(), value