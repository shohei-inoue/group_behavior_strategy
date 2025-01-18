import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from datetime import datetime

from envs.group_behavior.group_behavior_strategy_env import GroupBehaviorStrategyEnv
from agents.actor_critic_agent_ai import ActorCriticAgent
from models.actor_critic_model import ActorCriticModel
# from actor_critic_model import ActorCriticModel
# from agents.actor_critic_agent import ActorCriticAgent


# Define the test environment
def test_environment():
    # Initialize the custom environment
    env = GroupBehaviorStrategyEnv()
    
    # Extract observation and action space dimensions for the model
    hidden_size = 128  # Hidden layer size
    model = ActorCriticModel(
        hidden_size=hidden_size,
        follower_num=env.FOLLOWER_NUM,
        env_height=env.ENV_HEIGHT,
        env_width=env.ENV_WIDTH
        )
    optimizer = Adam(learning_rate=1e-3, clipnorm=1.0)
    agent = ActorCriticAgent(model, optimizer)
    # agent = ActorCriticAgent(
    #     env=env,
    #     model=model,
    #     optimizer=optimizer
    # )

    # Test environment and agent for a few episodes
    episodes = 1
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        print(f"Starting Episode {episode + 1}")
        
        for step_i in range(30):
            # Select an action
            action = agent.select_action(state)
            
            print(f"action: {action}")

            # Take a step in the environment
            next_state, reward, done, turncated, _ = env.step(action)
            
            # Train the agent
            agent.train_step(state, action, reward, next_state, done)
            # agent.update_model([state], [action], [reward])
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward

            print(f"Steps: {step_i + 1}")
        

        now = datetime.now()
        date_time_str = now.strftime("%Y%m%d_%H%M%S")
        # env.save_gif(episode + 1, date_time_str)

        
        print(f"Episode {episode + 1} finished with Total Reward: {total_reward}")

if __name__ == "__main__":
    test_environment()
