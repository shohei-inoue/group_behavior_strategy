from robots.red import Red

import gym
import numpy as np
import math
import sqlalchemy
import os
from dotenv import load_dotenv

# TODO 場所を変える必要あるかも
# .envファイルの読み込み
load_dotenv()

# データベース接続の設定
DATABASE_URL = os.getenv("DATABASE_URL")
engine = sqlalchemy.create_engine(DATABASE_URL)


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
  REWARD_DEFAULT            = -1    # デフォルト報酬 # TODO　これだと行ったり来たりするだけで報酬がもらえる
  REWARD_LEADER_COLLISION   = -100  # 衝突時の報酬
  REWARD_FOLLOWER_COLLISION = -1    # 追従者の衝突時の報酬
  REWARD_EXPLORATION        = 30     # 探査報酬
  # ----- finish parameter -----
  FINISH_EXPLORED_RATE = 0.95 # 探査率の終了条件
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
    """
    action:
      parameter B of the policy (continuity)    | (0 <= B <= 1): float
      parameter k_d of the policy (continuity)  | (0 <= k_d < inf): float
      parameter k_e of the policy (continuity)  | (0 <= k_e < inf): float
    """
    # actionの取得
    B = action["B"]
    k_d = action["k_d"]
    k_e = action["k_e"]
    
    # TODO 走行可能性の確率分布の生成
    # TODO 探査向上性の確率分布の生成
    # TODO 最終的な確率分布を生成

    dx = self.OUTER_BOUNDARY * np.cos(np.radians(theta))
    dy = self.OUTER_BOUNDARY * np.sin(np.radians(theta))

    self.agent_position, collision_flag = self.next_position(dy, dx)
    self.agent_trajectory.append(self.agent_position.copy())

    # TODO フォロワーの探査行動
    
    # 報酬計算
    reward = self.REWARD_DEFAULT
    if collision_flag:
      # リーダーが障害物に衝突した場合
      reward += self.REWARD_LEADER_COLLISION
    else:
      # 探査率の計算
      exploration_rate = self.explored_area / self.total_area
      if exploration_rate > self.previous_explored_area:
        # 探査率が上昇した場合
        reward += self.REWARD_EXPLORATION
    
    # TODO フォロワーの報酬計算

    # 終了条件
    turncated = False # TODO エピソードが途中で終了した場合
    done = self.explored_area >= self.total_area * self.FINISH_EXPLORED_RATE # 探査率が一定以上になった場合

    return self.agent_position, reward, done, turncated, {} # TODO infoを返す必要があるかも
  

  def next_position(self, dy, dx) -> tuple[np.ndarray, bool]:
    """
    障害物判定と次の移動先次を計算
    """
    # SAMPLING_NUM = 1000 # 軌跡線分のサンプリング数
    SAMPLING_NUM = max(150, int(np.ceil(np.linalg.norm([dy, dx]) * 10)))
    SAFE_DISTANCE = 1.0 # マップの安全距離
    collision_flag = False # 障害物判定フラグ

    for i in range(1, SAMPLING_NUM + 1):
      intermediate_position = np.array([
        self.agent_position[0] + (dy * i / SAMPLING_NUM),
        self.agent_position[1] + (dx * i / SAMPLING_NUM)
      ])

      # マップ内か判断
      if (0 < intermediate_position[0] < self.ENV_HEIGHT) and (0 < intermediate_position[1] < self.ENV_WIDTH):
        # サンプリング点が障害物でないか判断
        map_y = int(intermediate_position[0])
        map_x = int(intermediate_position[1])

        if self.map[map_y, map_x] == self.OBSTACLE_VALUE:  # 障害物
          print(f"Obstacle collided at : {intermediate_position}")
          collision_flag = True

          # 障害物に衝突する事前位置を計算
          collision_position = intermediate_position
          direction_vector = collision_position - self.agent_position
          norm_direction_vector = np.linalg.norm(direction_vector)

          # if norm_direction_vector > SAFE_DISTANCE:
          #     stop_position = self.agent_position + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)
          #     return stop_position
          # else:
          #     # 移動距離が安全距離より短い場合はそのまま停止
          #     return self.agent_position
          stop_position = self.agent_position + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)

        return stop_position, collision_flag
      
      else:
        continue

    return self.agent_position + np.array([dy, dx]), collision_flag

  
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
  

  def get_drivability(self):
    """
    走行可能性の確率分布を生成
    """
    for follower in self.follower_robots:
      pass
  

  