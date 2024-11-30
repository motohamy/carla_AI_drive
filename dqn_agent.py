import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class CarlaTrainer:
    def __init__(self, env, agent, log_dir='logs'):
        self.env = env
        self.agent = agent
        self.log_dir = log_dir
        self.rewards_history = []
        self.losses_history = []
        
    def train(self, num_episodes, max_steps=1000):
        for episode in tqdm(range(num_episodes)):
            state = self.env.reset()
            episode_reward = 0
            episode_losses = []
            
            for step in range(max_steps):
                # Select and perform action
                action = self.agent.select_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Store transition
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # Train agent
                if len(self.agent.memory) > self.agent.batch_size:
                    loss = self.agent.train()
                    episode_losses.append(loss)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            self.rewards_history.append(episode_reward)
            if episode_losses:
                self.losses_history.append(np.mean(episode_losses))
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.rewards_history[-10:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
                self._save_checkpoint(episode)
                
    def _save_checkpoint(self, episode):
        self.agent.save(f"{self.log_dir}/dqn_checkpoint_{episode}.pt")
        
    def plot_results(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.rewards_history)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.losses_history)
        plt.title('Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    # Initialize environment and agent
    env = CarlaEnvironment()  # From previous code
    state_dim = env.state_dim
    action_dim = 9  # 9 discrete actions mapped to 2D continuous space
    
    # Create DQN agent
    agent = DQNAgent(state_dim, action_dim)
    
    # Initialize trainer
    trainer = CarlaTrainer(env, agent)
    
    # Train
    num_episodes = 1000
    trainer.train(num_episodes)
    
    # Plot results
    trainer.plot_results()