from robots.red import Red

import gym
import numpy as np
import math


class GroupBehaviorStrategyEnv(gym.Env):
  """
  Description:
  A policy-on learning environment using Actor-Critic.
  """
  # ----- Metadata: rendering mode -----
  metadata = {'render.modes': ['human', 'rgb_array']}
  # ----- size of drawing -----
  ENV_WIDTH = 150
  ENV_HEIGHT = 60
  # ----- exploration parameter -----
  OUTER_BOUNDARY  = 10.0  # 探査半径(外側)
  INNER_BOUNDARY  = 0.0   # 探査半径(内側)
  MEAN            = 0.0   # 探査中心の平均
  VARIANCE        = 10.0  # 探査中心の分散
  OBSTACLE_VALUE  = 1000  # 障害物値
  # ----- reward parameter -----
  REWARD_DEFAULT            = 50    # デフォルト報酬 # TODO　これだと行ったり来たりするだけで報酬がもらえる
  REWARD_LEADER_COLLISION   = -100  # 衝突時の報酬
  REWARD_FOLLOWER_COLLISION = -1    # 追従者の衝突時の報酬
  # ----- initial parameter -----
  INITIAL_B     = 0.5 # B_0
  INITIAL_K_D   = 1.0 # k_d_0
  INITIAL_K_E   = 1.0 # k_e_0
  INIT_POSITION = [10.0, 10.0] # 初期位置(y, x)
  # ----- follower parameter -----
  FOLLOWER_NUM             = 10 # 追従者の数
  FOLLOWER_POSITION_OFFSET = 5.0 # 追従者の初期位置のオフセット


  def __init__(self) -> None:
    """
    constructor:
      action_space:
        parameter B of the policy (continuity)    | (0 <= B <= 1): float
        parameter k_d of the policy (continuity)  | (0 <= k_d < inf): float
        parameter k_e of the policy (continuity)  | (0 <= k_e < inf): float
      observation_space:
        follower_collision_info_list                      | (lits[(0 < mean < 2 *pi), (0 <= variance < inf)]): list[[mean: float | None, variance: float | None], ...] 
        leader_collision_point: [y, x]                    | (0 <= y < ENV_HIGHT, 0 <= x < ENV_WIDTH): list[float, float]
        parameter B of the policy (continuity, current)   | (0 <= B <= 1): float
        parameter k_d of the policy (continuity, current) | (0 <= k_d < inf): float 
        parameter k_e of the policy (continuity, current) | (0 <= k_e < inf): float 
      reward_range:
        -inf ~ inf (仮)
    """
    super().__init__()
    # ----- action_space -----
    B_space = gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)

    # parameter k_d
    k_d_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32)

    # parameter k_e
    k_e_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32)

    self.action_space = gym.spaces.Dict({
        "B": B_space,
        "k_d": k_d_space,
        "k_e": k_e_space
    })

    # ----- observation_space -----
    # follower_collision_info_list
    follower_collision_info_template = gym.spaces.Dict({
      "mean": gym.spaces.Dict({
          "value": gym.spaces.Box(low=0.0, high=2 * np.pi, shape=(), dtype=np.float32),  # mean in range (0, 2*pi)
          "is_none": gym.spaces.Discrete(2)  # 0 if not None, 1 if None
      }),
      "variance": gym.spaces.Dict({
          "value": gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),  # variance >= 0
          "is_none": gym.spaces.Discrete(2)  # 0 if not None, 1 if None
      })
    })
    follower_collision_info_space = gym.spaces.Tuple([follower_collision_info_template] * self.FOLLOWER_NUM)

    # leader_collision_point
    leader_low = np.array([0.0, 0.0])  # y >= 0, x >= 0
    leader_high = np.array([self.ENV_HEIGHT, self.ENV_WIDTH])  # y < ENV_HEIGHT, x < ENV_WIDTH
    leader_collision_point_space = gym.spaces.Box(low=leader_low, high=leader_high, shape=(2,), dtype=np.float32)

    # Combine all spaces into a dictionary space
    self.observation_space = gym.spaces.Dict({
        "follower_collision_info_list": follower_collision_info_space,
        "leader_collision_point": leader_collision_point_space,
        "B": B_space,
        "k_d": k_d_space,
        "k_e": k_e_space
    })

    # ----- reward_range -----
    self.reward_range = (-np.inf, np.inf) # TODO 仮置き

    # ----- set initial parameter -----
    self.map = np.zeros((self.ENV_HEIGHT, self.ENV_WIDTH)) # マップの初期化
    self.generate_obstacles() # 障害物の生成
    self.explored_map = np.zeros((self.ENV_HEIGHT, self.ENV_WIDTH)) # 探査済みマップの初期化
    self.total_area   = np.prod(self.map.shape) - np.sum(self.map == self.OBSTACLE_VALUE) # 探査可能なエリア
    self.explored_area = 0 # 探査済みエリア
    self.previous_explored_area = 0.0 # 1ステップ前の探査率

    self.agent_position   = self.INIT_POSITION # 位置情報の初期化
    self.agent_trajectory = [self.agent_position.copy()] # 軌跡の初期化
    self.env_frames       = [] # 描画用のフレームの初期化

    # フォロワーの追加
    self.follower_robots = [Red(
        id=f'follower_{index}',
        env=self,
        agent_position=self.agent_position,
        x=self.agent_position[1] + self.FOLLOWER_POSITION_OFFSET * math.cos((2 * math.pi * index / (self.FOLLOWER_NUM))),
        y=self.agent_position[0] + self.FOLLOWER_POSITION_OFFSET * math.sin((2 * math.pi * index / (self.FOLLOWER_NUM))),
    ) for index in range(self.FOLLOWER_NUM)] 


  def _reset(self) -> None:
    """
    initialize the environment
    """
    self.agent_position = self.INIT_POSITION # 位置情報の初期化
    self.agent_trajectory = [self.agent_position.copy()] # 軌跡の初期化
    self.explored_area = 0 # 探査済みエリアの初期化
    self.explored_map.fill(0) # 探査済みマップの初期化
    self.previous_explored_area = 0.0 # 1ステップ前の探査率の初期化
    self.follower_robots = [Red(
        id=f'follower_{index}',
        env=self,
        agent_position=self.agent_position,
        x=self.agent_position[1] + self.FOLLOWER_POSITION_OFFSET * math.cos((2 * math.pi * index / (self.FOLLOWER_NUM))),
        y=self.agent_position[0] + self.FOLLOWER_POSITION_OFFSET * math.sin((2 * math.pi * index / (self.FOLLOWER_NUM))),
    ) for index in range(self.FOLLOWER_NUM)] # フォロワーの初期化


  def _render(self, mode='human'):
    if mode == 'rgb_array':
      # return np.array([])
      pass
    elif mode == 'human':
      pass
    else:
      # super().render(mode=mode)
      pass

  def _step(self, action):
    # return super().step(action)
    pass

  
  def _close(self):
    pass

  def _seed(self, seed=None):
    pass
  

  def generate_obstacles(self) -> None:
    """
    障害物の生成
    """
    # 壁の生成
    self.map[0, :]                    = self.OBSTACLE_VALUE # 上辺
    self.map[self.ENV_HEIGHT - 1, :]  = self.OBSTACLE_VALUE # 下辺
    self.map[:, 0]                    = self.OBSTACLE_VALUE # 左辺
    self.map[:, self.ENV_WIDTH - 1]   = self.OBSTACLE_VALUE # 右辺

    # 他の障害物の生成
    self.map[20, :20]     = self.OBSTACLE_VALUE
    self.map[30:, 40]     = self.OBSTACLE_VALUE
    self.map[:40, 70]     = self.OBSTACLE_VALUE
    self.map[20:40, 100]  = self.OBSTACLE_VALUE