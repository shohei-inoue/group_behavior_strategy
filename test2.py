from envs.group_behavior.group_behavior_strategy_env import GroupBehaviorStrategyEnv
from agents.actor_critic_agent import ActorCriticAgent
from models.actor_critic_model import ActorCriticModel
from tensorflow.keras.optimizers import Adam

def main(episodes=10, steps=100) -> None:
    """
    学習済みのモデルを使って行動を選択する
    """
    env = GroupBehaviorStrategyEnv() 
    model = ActorCriticModel(
       follower_num=env.FOLLOWER_NUM,
       env_height=env.ENV_HEIGHT,
       env_width=env.ENV_WIDTH,
       hidden_size=128
       )
    
    agent = ActorCriticAgent(
       env=env,
       model=model,
       optimizer=Adam(learning_rate=0.001)
    )

    # episode loop
    for episode in range(episodes):
      state = agent.env.reset() # 環境の初期化
      done = False
      total_reward = 0

      # step loop
      for step in range(steps):
        action = agent.select_action(state) # 行動選択
        next_state, reward, done, turncated, info = agent.step(action) # 行動実行
        total_reward += reward
        
        # 経験を蓄積しモデルを更新
        agent.update_model([state], [action], [reward])
        state = next_state

        # 終了条件
        if done:
          break
        
        print(f"agent step: {step}")
    
      print(f"Episode {episode}: Total Reward = {total_reward}")
    
    agent.save("actor_critic_model.h5")


if __name__ == "__main__":
   main(episodes=5, steps=10)
    
    