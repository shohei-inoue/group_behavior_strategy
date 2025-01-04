import tensorflow as tf
from tensorflow.keras import layers

class ActorCriticModel(tf.keras.Model):
  def __init__(
      self, 
      follower_num,
      env_height,
      env_width,
      hidden_size=128
      ):
    """
    TODO
    constructor: initialize the model
    """
    super(ActorCriticModel, self).__init__()

    # parameters
    self.follower_num = follower_num
    self.env_height   = env_height
    self.env_width    = env_width

    # 共有ネットワーク(観測空間の特徴量を抽出するためのネットワーク)
    self.shared = tf.keras.Sequential([
      layers.Flatten(), # 観測を平坦化
      layers.Dense(hidden_size, activation='relu'),
      layers.Dense(hidden_size, activation='relu')
    ])

    # actor network
    self.actor_B    = layers.Dense(1, activation='sigmoid') # 0 <= B <= 1
    self.actor_k_d  = layers.Dense(1, activation='softplus') # 0 <= k_d < inf
    self.actor_k_e  = layers.Dense(1, activation='softplus') # 0 <= k_e < inf 

    # critic network
    self.critic = layers.Dense(1) # 状態価値


  def call(self, inputs):
    """
    TODO
    inputs: 観測空間のデータ
    """
    follower_info = tf.reshape(
      inputs['follower_collision_info_list'],
      (inputs['followers_collision_info_list'].shape[0], -1)
    )

    # リーダーの衝突点とポリシーの現在値を結合
    combined_inputs = tf.concat([
      follower_info,
      inputs['leader_collision_point'],
      inputs['B'],
      inputs['k_d'],
      inputs['k_e']
    ])

    x = self.shared(combined_inputs)

    # Actorの出力
    B   = self.actor_B(x)
    k_d = self.actor_k_d(x)
    k_e = self.actor_k_e(x)
    action_probs = tf.concat([B, k_d, k_e], axis=-1)

    # criticの出力
    state_values = self.critic(x)

    return action_probs, state_values
  

  def compute_loss(self, action_probs, state_values, actions, rewards, done):
    """
    TODO
    action_probs: 行動の確率
    state_values: 状態価値
    actions: 行動
    rewards: 報酬
    done: 終了フラグ
    """
    # criticの損失
    pass

    # actorの損失
    pass

    # return total_loss