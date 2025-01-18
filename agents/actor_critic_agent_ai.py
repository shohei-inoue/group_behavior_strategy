import numpy as np
import tensorflow as tf
import gym

class ActorCriticAgent:
    def __init__(self, model, optimizer, gamma=0.99):
        """
        Initialize the Actor-Critic agent.

        Args:
            model (ActorCriticModel): The Actor-Critic model.
            optimizer (tf.keras.optimizers.Optimizer): Optimizer for training.
            gamma (float): Discount factor for future rewards.
        """
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state):
      """
      Select an action based on the current state.

      Args:
          state (dict): Current state of the environment.

      Returns:
          action (dict): Selected action.
      """
      # フラット化処理
      follower_collision_info = np.array([
        [entry["mean"], entry["covariance"], entry["has_collisions"], entry["count"]]
        for entry in state["follower_collision_info_list"]
      ], dtype=np.float32).flatten()

      leader_collision_point = state["leader_collision_point"]["value"]

      # 状態をテンソル形式に変換
      state_tensor = {
          "follower_collision_info_list": tf.convert_to_tensor(np.expand_dims(follower_collision_info, axis=0)),
          "leader_collision_point": tf.convert_to_tensor(np.expand_dims(leader_collision_point, axis=0)),
          "B": tf.convert_to_tensor([[state["B"]]], dtype=tf.float32),
          "k_d": tf.convert_to_tensor([[state["k_d"]]], dtype=tf.float32),
          "k_e": tf.convert_to_tensor([[state["k_e"]]], dtype=tf.float32),
          "k_c": tf.convert_to_tensor([[state["k_c"]]], dtype=tf.float32),
      }

      # state_tensor = tf.expand_dims(state_tensor, axis=0)
      # アクションと状態価値を計算
      actions, _ = self.model(state_tensor)

      print(f"select_action:{actions}")

      # アクション辞書を生成して返す
      return {key: value.numpy()[0][0] for key, value in actions.items()}


    def compute_returns(self, rewards, dones):
        """
        Compute discounted returns for a trajectory.

        Args:
            rewards (list): List of rewards for the trajectory.
            dones (list): List of done flags for the trajectory.

        Returns:
            list: Discounted returns.
        """
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return returns

    def train_step(self, state, action, reward, next_state, done):
        """
        Perform a single training step.

        Args:
            state (dict): Current state.
            action (dict): Taken action.
            reward (float): Received reward.
            next_state (dict): Next state.
            done (bool): Whether the episode has ended.

        """
        with tf.GradientTape() as tape:
            # Convert states and actions to tensors
            print(state.items())
            state_tensor = {}
            for key, value in state.items():
                if key == "follower_collision_info_list":
                    # 各フォロワーの情報を平坦化
                    flattened = np.array([
                        [entry['mean'], entry['covariance'], entry['count'], entry['has_collisions']]
                        for entry in value
                    ], dtype=np.float32).flatten()
                    state_tensor[key] = tf.convert_to_tensor([flattened], dtype=tf.float32)
                elif key == "leader_collision_point":
                    # leader_collision_pointをテンソル化
                    state_tensor[key] = tf.convert_to_tensor([value['value']], dtype=tf.float32)
                elif isinstance(value, (int, float, np.float32)):
                    state_tensor[key] = tf.convert_to_tensor([value], dtype=tf.float32)
                else:
                  raise ValueError(f"Unsupported value type for key: {key}, type: {type(value)}")

            next_state_tensor = {}
            for key, value in next_state.items():
                if key == "follower_collision_info_list":
                  # 各フォロワーの情報を平坦化
                  flattened = np.array([
                      [entry['mean'], entry['covariance'], entry['count'], entry['has_collisions']]
                      for entry in value
                  ], dtype=np.float32).flatten()
                  next_state_tensor[key] = tf.convert_to_tensor([flattened], dtype=tf.float32)
                elif key == "leader_collision_point":
                    # leader_collision_pointをテンソル化
                    next_state_tensor[key] = tf.convert_to_tensor([value['value']], dtype=tf.float32)
                elif isinstance(value, (int, float, np.float32)):
                    next_state_tensor[key] = tf.convert_to_tensor([value], dtype=tf.float32)
                else:
                    raise ValueError(f"Unsupported value type for key: {key}, type: {type(value)}")

            # Get predictions from the model
            actions, value = self.model(state_tensor)
            _, next_value = self.model(next_state_tensor)

            # Compute the target value
            target = reward + (1 - done) * self.gamma * next_value
            target = tf.stop_gradient(target)

            # Compute the critic loss
            critic_loss = tf.square(target - value)

            # Compute the actor loss
            actor_loss = 0
            for key in action.keys():
                log_prob = tf.math.log(actions[key] + 1e-8)  # Avoid log(0)
                actor_loss += -log_prob * (target - value)

            # Combine losses
            total_loss = tf.reduce_mean(critic_loss + actor_loss)

        # Backpropagate and update the model
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    

    def preprocess_state(state):
      state_tensor = {}
      for key, value in state.items():
          if isinstance(value, dict):  # 辞書の場合
              if "value" in value:  # `value` キーを含む
                  state_tensor[key] = tf.convert_to_tensor(np.expand_dims(value["value"], axis=0), dtype=tf.float32)
              else:
                  raise ValueError(f"Unsupported dictionary structure for key: {key}")
          elif isinstance(value, tuple):  # タプルの場合
              flattened = np.array([
                  [entry["mean"], entry["covariance"], entry["has_collisions"], entry["count"]]
                  for entry in value
              ], dtype=np.float32).flatten()
              state_tensor[key] = tf.convert_to_tensor(np.expand_dims(flattened, axis=0), dtype=tf.float32)
          else:  # その他の値
              state_tensor[key] = tf.convert_to_tensor(np.expand_dims(value, axis=0), dtype=tf.float32)
      return state_tensor
    

    def train(self, env, episodes):
        """
        Train the agent in the given environment.

        Args:
            env (gym.Env): Environment to train in.
            episodes (int): Number of episodes to train.
        """
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.train_step(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
