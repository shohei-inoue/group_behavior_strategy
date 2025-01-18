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
    self.actor_k_c  = layers.Dense(1, activation='softplus') # 0 <= k_c < inf 

    # critic network
    self.critic = layers.Dense(1) # 状態価値


  def call(self, inputs):
    """
    TODO
    inputs: 観測空間のデータ
    """
    # follower_collision_info_list をそのまま使用
    follower_info = inputs['follower_collision_info_list']  # (1, 40)

    # 他の入力を (バッチサイズ, 1) に整形
    B = tf.reshape(inputs['B'], shape=(-1, 1))  # (1, 1)
    k_d = tf.reshape(inputs['k_d'], shape=(-1, 1))  # (1, 1)
    k_e = tf.reshape(inputs['k_e'], shape=(-1, 1))  # (1, 1)
    k_c = tf.reshape(inputs['k_c'], shape=(-1, 1))  # (1, 1)

    # leader_collision_point をそのまま使用
    # leader_info = inputs['leader_collision_point']  # (1, 2)
    leader_info = tf.reshape(inputs['leader_collision_point'], shape=(-1, 1))

    # 全ての情報を結合
    combined_inputs = tf.concat([
        follower_info,  # (1, 40)
        leader_info,    # (1, 1)
        B,              # (1, 1)
        k_d,            # (1, 1)
        k_e,            # (1, 1)
        k_c             # (1, 1)
    ], axis=-1)  # 結合後の形状は (1, 45)


    # print(f"B: {B}")
    # print(f"k_d: {k_d}")
    # print(f"k_e: {k_e}")
    # print(f"k_c: {k_c}")

    # print(f"inputs: {inputs}")

    # print(f"Combined Inputs: {combined_inputs}")
    # print(f"Contains NaN: {tf.reduce_any(tf.math.is_nan(combined_inputs))}")
    # print(f"Max Value: {tf.reduce_max(combined_inputs)}")
    # print(f"Min Value: {tf.reduce_min(combined_inputs)}")

    features = self.shared(combined_inputs)

    # print(f"features: {features}")

    # Actorの出力
    B   = self.actor_B(features)
    k_d = self.actor_k_d(features)
    k_e = self.actor_k_e(features)
    k_c = self.actor_k_c(features)

    # print(f"B: {B}")
    # print(f"k_d: {k_d}")
    # print(f"k_e: {k_e}")
    # print(f"k_c: {k_c}")


    actions = {
      'B': B,
      'k_d': k_d,
      'k_e': k_e,
      'k_c': k_c
    }

    # criticの出力
    state_values = self.critic(features)

    return actions, state_values
  

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