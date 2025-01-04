import tensorflow as tf
import numpy as np

class EpsilonGreedyAgent:
  def __init__(
      self, 
      env, 
      model, 
      gamma=0.99,
      epsilon=1.0,
      ):
    """
    constructor: initialize the agent
    """
    super(EpsilonGreedyAgent, self).__init__()
    self.env            = env # environment
    self.model          = model # learning model
    self.gamma          = gamma # 割引率
    self.epsilon        = epsilon # epsilon-greedy法のεの初期値
    self.epsilon_min    = 0.01 # εの最小値
    self.epsilon_decay  = 0.995 # εの減衰率
    self.learning_rate  = 0.001 # 学習率
    self.optimizer      = tf.keras.optimizers.Adam(self.learning_rate) # optimizer
    self.state         = None # 状態
    self.action        = None
    self.reward        = None
    self.terncalated   = False # 終了フラグ
    self.done          = False # 終了フラグ
  

  def policy(self):
    """
    TODO
    policy: 行動選択
    epsilon-greedy法を使って行動を選択する
    """
    if np.random.rand() <= self.epsilon:
      print('random')
      return np.random.choice(self.env.action_space)
    else:
      print('greedy')
      # return np.argmax(self.model.call(input))
  

  def run(self, episodes=10, steps=100) -> None:
    """
    学習済みのモデルを使って行動を選択する
    """
    # episode loop
    for episode in range(episodes):
      state = self.env.reset() # 環境の初期化

      # step loop
      for step in range(steps):
        action = self.policy() # 行動選択
        next_state, reward, done, turncated, info = self.env.step(action) # 行動実行
        td_error = self.log(state, action, reward, next_state, done) # ログ
        self.learn(td_error) # TODO 学習
        
        # 終了条件
        if done:
          break

  def learn(self, td_error):
    """
    TODO
    学習
    """
    pass
    # with tf.GradientTape() as tape:
    #   action = self.model.call(state)
    #   loss = self.loss(action, td_error)
    # grads = tape.gradient(loss, self.model.trainable_variables)
    # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
  

  def log(self, state, action, reward, next_state, done):
    """
    TODO
    state: dict
    action: dict
    reward: float
    next_state: dict
    done: bool
    """
    pass
    # state_value = self.critic(state)
    # next_state_value = self.critic(next_state)
    # td_target = reward + self.gamma * next_state_value * (1 - done)
    # td_error = td_target - state_value
    # return td_error
  
