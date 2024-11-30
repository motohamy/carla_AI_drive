from carla import CarlaEnvironment
from base_agent import DQNAgent
from dqn_agent import CarlaTrainer
import os

def main():
    # Create save directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize environment
    env = CarlaEnvironment()
    
    # Initialize agent
    state_dim = env.state_dim
    action_dim = 9  # 9 discrete actions for DQN
    agent = DQNAgent(state_dim, action_dim)
    
    # Initialize trainer
    trainer = CarlaTrainer(env, agent)
    
    # Train
    num_episodes = 1000
    trainer.train(num_episodes)
    
    # Plot results
    trainer.plot_results()
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main()