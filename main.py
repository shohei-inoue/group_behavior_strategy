from envs.group_behavior.group_behavior_strategy_env import GroupBehaviorStrategyEnv
from agents.actor_critic_agent import EpsilonGreedyAgent
from models.actor_critic_model import ActorCriticModel
import numpy as np


def main():
  # environment
  env = GroupBehaviorStrategyEnv()

  # model
  model = ActorCriticModel(
    follower_num=env.FOLLOWER_NUM,
    env_height=env.ENV_HEIGHT,
    env_width=env.ENV_WIDTH
  )

  # agent
  agent = EpsilonGreedyAgent(
    env=env,
    model=model,
    gamma=0.99,
    epsilon=0.2
  )

  # run
  try:
    agent.run()
  except Exception as e:
    print(f"Error during agent run:{e}")


if __name__ == '__main__':
  main()