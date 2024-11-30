import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class PPOActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and standard deviation for continuous actions
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = self.network(state)
        mean = torch.tanh(self.mean(x))  # Bound mean to [-1, 1]
        log_std = torch.clamp(self.log_std(x), -20, 2)  # Bound log_std for stability
        return mean, log_std

class PPOCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(PPOCritic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        
    def add(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def get_batch(self):
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        rewards = torch.FloatTensor(np.array(self.rewards))
        next_states = torch.FloatTensor(np.array(self.next_states))
        dones = torch.FloatTensor(np.array(self.dones))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        values = torch.FloatTensor(np.array(self.values))
        
        return states, actions, rewards, next_states, dones, old_log_probs, values

class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor = PPOActor(state_dim, action_dim).to(self.device)
        self.critic = PPOCritic(state_dim).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Initialize memory
        self.memory = PPOMemory()
        
        # PPO parameters
        self.clip_param = 0.2
        self.ppo_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mean, log_std = self.actor(state)
            std = log_std.exp()
            
            if evaluate:
                action = mean
            else:
                normal = Normal(mean, std)
                action = normal.sample()
                action = torch.clamp(action, -1, 1)
                
            value = self.critic(state)
            if not evaluate:
                normal = Normal(mean, std)
                log_prob = normal.log_prob(action).sum(-1)
                return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()
            
            return action.cpu().numpy()[0]
    
    def train(self):
        # Get batch data
        states, actions, rewards, next_states, dones, old_log_probs, values = self.memory.get_batch()
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        
        # Compute advantages
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = self.critic(next_states[-1].unsqueeze(0).to(self.device)).detach()
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Compute current log probs and values
            mean, log_std = self.actor(states)
            std = log_std.exp()
            normal = Normal(mean, std)
            current_log_probs = normal.log_prob(actions).sum(-1)
            entropy = normal.entropy().mean()
            current_values = self.critic(states).squeeze()
            
            # Compute ratios and surrogate losses
            ratios = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            
            # Compute losses
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - current_values).pow(2).mean()
            
            # Total loss
            loss = actor_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        # Clear memory after update
        self.memory.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        self.memory.add(state, action, reward, next_state, done, log_prob, value)
    
    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])