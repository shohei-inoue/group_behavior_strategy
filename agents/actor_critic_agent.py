import tensorflow as tf
import numpy as np

class ActorCriticAgent:
  def __init__(
      self, 
      env, 
      model, 
      optimizer,
      gamma=0.99,
      learning_rate=1e-3
      ):
    """
    constructor: initialize the agent
    """
    super(ActorCriticAgent, self).__init__()
    self.env            = env # environment
    self.model          = model # learning model
    self.gamma          = gamma # 割引率
    self.learning_rate  = learning_rate # 学習率
    self.optimizer      = optimizer
    self.state          = self.env.reset() # 状態
  

  def select_action(self, state):
    """
    状態に基づき行動を選択
    """
    state = np.expand_dims(state, axis=0)
    logits, _ = self.model(state, axis=0)
    action_prob = tf.nn.softmax(logits)
    action = np.random.choice(len(action_prob[0]), p=action_prob.numpy()[0])

  
  def step(self, action):
    """
    環境とのインタラクション
    """
    next_state, reward, done, turncated, info = self.env.step(action)
    return next_state, reward, done, turncated, info


  def update_model(self, states, actions, rewards, next_states, dones):
    """
    モデルの更新
    """
    with tf.GradientTape() as tape:
      # 状態価値の計算
      _, values = self.model(np.array(states))
      _, next_values = self.model(np.array(next_states))

      # advantageの計算 TODO ここはポリシー勾配法？
      targets = np.array(rewards) + self.gamma * np.array(next_values) * (1 - np.array(dones))
      advantages = targets - values

      # Actor損失
      logits, _ = self.model(np.array(states))
      action_masks = tf.one_hot(actions, logits.shape[-1]) # 行動のone-hot表現
      log_probs = tf.reduce_sum(action_masks * tf.nn.log_softmax(logits), axis=1)
      actor_loss = -tf.reduce_mean(log_probs * advantages)

      # Critic損失
      critic_loss = tf.reduce_mean(tf.square(advantages))

      # 合計損失
      loss = actor_loss + critic_loss

    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
  

  def reset(self):
    """
    環境をリセット
    """
    self.state = self.env.reset()
  

  def save(self, file_path):
    """
    モデルの保存
    """
    self.model.save_weights(file_path)
  

  def load(self, file_path):
    """
    モデルのロード
    """
    self.model.load_weights(file_path)
  
