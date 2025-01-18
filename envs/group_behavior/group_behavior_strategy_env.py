from robots.red import Red

import gym
import numpy as np
import math
import sqlalchemy
import os
import imageio
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
from dotenv import load_dotenv
from scipy.stats import vonmises

# TODO 場所を変える必要あるかも
# .envファイルの読み込み
load_dotenv()

# データベース接続の設定
# DATABASE_URL = os.getenv("DATABASE_URL")
# engine = sqlalchemy.create_engine(DATABASE_URL)


class GroupBehaviorStrategyEnv(gym.Env):
  """
  Description:
  A policy-on learning environment using Actor-Critic.
  """
  # ----- Metadata: rendering mode -----
  metadata = {'render.modes': ['human', 'rgb_array']}
  # ----- save data parameter -----
  SAVE_FRAMES = True # フレームの保存
  # ----- size of drawing -----
  ENV_WIDTH = 150
  ENV_HEIGHT = 60
  # ----- exploration parameter -----
  OUTER_BOUNDARY  = 10.0  # 探査半径(外側)
  INNER_BOUNDARY  = 0.0   # 探査半径(内側)
  MEAN            = 0.0   # 探査中心の平均
  VARIANCE        = 10.0  # 探査中心の分散
  OBSTACLE_VALUE  = 1000  # 障害物値
  FOLLOWER_STEP   = 100   # フォロワーのステップ数
  # ----- reward parameter -----
  REWARD_DEFAULT            = -1    # デフォルト報酬
  REWARD_LEADER_COLLISION   = -100  # 衝突時の報酬
  REWARD_FOLLOWER_COLLISION = -1    # 追従者の衝突時の報酬
  REWARD_EXPLORATION        = 50     # 探査報酬
  REWARD_FINISH             = 100
  # ----- finish parameter -----
  FINISH_EXPLORED_RATE = 0.95 # 探査率の終了条件
  # ----- Von Mises distribution parameter -----
  KAPPA = 1.0 # 逆温度
  ANGLES = np.linspace(0, 2 * np.pi, 360) # 角度
  # ----- initial parameter -----
  INITIAL_B     = 0.5 # B_0
  INITIAL_K_D   = 1.0 # k_d_0
  INITIAL_K_E   = 1.0 # k_e_0
  INITIAL_K_C   = 1.0 # k_c_0
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
        parameter k_c of the policy (continuity)  | (0 <= k_c < inf): float
      observation_space:
        follower_collision_info_list                      | (lits[(0 < mean < 2 *pi), (0 <= covariance < inf)]): list[[mean: float | None, covariance: float | None], ...] 
        leader_collision_point: theta                     | (0 <= theta < 2 * pi: float | None
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

    # parameter k_c
    k_c_space = gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32)

    self.action_space = gym.spaces.Dict({
      "B": B_space,
      "k_d": k_d_space,
      "k_e": k_e_space,
      "k_c": k_c_space
    })

    # ----- observation_space -----
    follower_collision_info_template = gym.spaces.Dict({
      "has_collisions": gym.spaces.Discrete(2),  # 0: 障害物なし, 1: 障害物あり
      "mean": gym.spaces.Box(low=0.0, high=2 * np.pi, shape=(), dtype=np.float32),  # mean in range (0, 2*pi)
      "covariance": gym.spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float32),  # covariance >= 0
      "count": gym.spaces.Box(low=0, high=np.inf, shape=(), dtype=np.int32)  # 衝突回数
    })
    follower_collision_info_space = gym.spaces.Tuple([follower_collision_info_template] * self.FOLLOWER_NUM)

    # leader_collision_point
    leader_collision_point_space = gym.spaces.Dict({
      "value": gym.spaces.Box(low=0.0, high=2 * np.pi, shape=(), dtype=np.float32),  # 値範囲
      "is_none": gym.spaces.Discrete(2)  # 0 if not None, 1 if None
    })

    # Combine all spaces into a dictionary space
    self.observation_space = gym.spaces.Dict({
      "follower_collision_info_list": follower_collision_info_space,
      "leader_collision_point": leader_collision_point_space,
      "B": B_space,
      "k_d": k_d_space,
      "k_e": k_e_space,
      "k_c": k_c_space,
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
    self.agent_step = 0 # エージェントのステップ

    self.agent_position                 = self.INIT_POSITION # 位置情報の初期化
    self.previous_agent_position        = None # 1ステップ前の位置情報
    self.agent_trajectory               = [self.agent_position.copy()] # 軌跡の初期化
    self.drivability_pdf_list           = [] # 走行可能性の確率分布の初期化(描画で利用)
    self.explore_improvement_pdf_list   = [] # 探査向上性の確率分布の初期化(描画で利用)
    self.result_pdf_list                = [] # 結合確率分布の初期化(描画で利用)
    self.env_frames                     = [] # 描画用のフレームの初期化

    # フォロワーの追加
    self.follower_robots = [Red(
      id=f'follower_{index}',
      env=self,
      agent_position=self.agent_position,
      x=self.agent_position[1] + self.FOLLOWER_POSITION_OFFSET * math.cos((2 * math.pi * index / (self.FOLLOWER_NUM))),
        y=self.agent_position[0] + self.FOLLOWER_POSITION_OFFSET * math.sin((2 * math.pi * index / (self.FOLLOWER_NUM))),
    ) for index in range(self.FOLLOWER_NUM)] 

    # leader_collision_point を None に設定
    leader_collision_point = {
      "value": 0.0,
      "is_none": 1  # None を示すフラグ
    }

    # follower_collision_info_list の各要素を None に設定
    follower_collision_info_list = tuple(
      {
        "has_collisions": 0, 
        "mean": 0.0,  
        "covariance": 0.0,  
        "count": 0
      }
      for _ in range(self.FOLLOWER_NUM)
    )

    # 初期状態を生成
    self.state = {
        "follower_collision_info_list": follower_collision_info_list,
        "leader_collision_point": leader_collision_point,
        "B": self.INITIAL_B,
        "k_d": self.INITIAL_K_D,
        "k_e": self.INITIAL_K_E,
        "k_c": self.INITIAL_K_C,
    }


  def reset(self):
    """
    initialize the environment
    """
    self.agent_position = self.INIT_POSITION # 位置情報の初期化
    self.agent_trajectory = [self.agent_position.copy()] # 軌跡の初期化
    self.previous_agent_position = None # 1ステップ前の位置情報
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
    self.result_pdf_list = [] # 結合確率分布の初期化
    self.drivability_pdf_list = [] # 走行可能性の確率分布の初期化
    self.explore_improvement_pdf_list = [] # 探査向上性の確率分布の初期化
    self.agent_step = 0

    # leader_collision_point を None に設定
    leader_collision_point = {
      "value": 0.0,  # ダミー値 (使用されない)
      "is_none": 1  # None を示すフラグ
    }

    # follower_collision_info_list の各要素を None に設定
    follower_collision_info_list = tuple(
      {
        "has_collisions": 0,
        "mean": 0.0,
        "covariance": 0.0,
        "count": 0,
      }
      for _ in range(self.FOLLOWER_NUM)
    )

    # 初期状態を生成
    self.state = {
        "follower_collision_info_list": follower_collision_info_list,
        "leader_collision_point": leader_collision_point,
        "B": self.INITIAL_B,
        "k_d": self.INITIAL_K_D,
        "k_e": self.INITIAL_K_E,
        "k_c": self.INITIAL_K_C,
    }

    return self.state


  def render(self, save_frames = False, mode='human'):
    """
    render the environment gridspecを利用
    マップの描画
    フォロワーの描画
    リーダーの描画
    """
    if mode == 'rgb_array':
      pass
    elif mode == 'human':
      fig = plt.figure("Environment", figsize=(12, 12))
      fig.clf()

      # 全体のグリッドを設定
      gs_master = gridspec.GridSpec(
        nrows=3,
        ncols=3,
        height_ratios=[1, 1, 1],
      )

      # マップのグリッドを設定
      gs_env = gridspec.GridSpecFromSubplotSpec(
        nrows=1,
        ncols=3,
        subplot_spec=gs_master[0, :],
      )

      # gs_env を全体を1つのプロットとして設定
      ax_env = fig.add_subplot(gs_env[:, :])
      ax_env.set_title("Environment Map")
      ax_env.set_xlabel("X-axis")
      ax_env.set_ylabel("Y-axis")

      # ---------------------------------------- Environment ----------------------------------------
      # 環境のマップを描画
      cmap = mcolors.ListedColormap(['white', 'gray', 'black'])
      bounds = [0, 1, self.OBSTACLE_VALUE, self.OBSTACLE_VALUE + 1]
      norm = mcolors.BoundaryNorm(bounds, cmap.N)

      # マップの描画
      ax_env.imshow(
        self.map,
        cmap='gray_r',
        origin='lower',
        extent=[0, self.ENV_WIDTH, 0, self.ENV_HEIGHT],
      )
      # 探査済みマップの描画
      ax_env.imshow(
        self.explored_map,
        cmap=cmap,
        alpha=0.5,
        norm=norm,
        origin='lower',
        extent=[0, self.ENV_WIDTH, 0, self.ENV_HEIGHT],
      )
      # 探査中心位置の描画
      ax_env.scatter(
        x=self.agent_position[1],
        y=self.agent_position[0],
        color='blue',
        s=100,
        label='Explore Center',
      )
      # 軌跡の描画
      trajectory = np.array(self.agent_trajectory, dtype=np.float32)

      ax_env.plot(
        # self.agent_trajectory[:, 1],
        # self.agent_trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 0],
        color='blue',
        linewidth=2,
        label='Explore Center Trajectory',
      )
      # 探査領域の描画
      explore_area = Circle(
        (self.agent_position[1], self.agent_position[0]),
        self.OUTER_BOUNDARY,
        color='black',
        fill=False,
        linewidth=1,
        label='Explore Area',
      )
      ax_env.add_patch(explore_area)
      # Followerの描画
      for follower in self.follower_robots:
        ax_env.scatter(
          x=follower.data['x'].iloc[-1],
          y=follower.data['y'].iloc[-1],
          color='red',
          s=10,
          # label=follower.id,
        )
        ax_env.plot(
          follower.data['x'],
          follower.data['y'],
          color='gray',
          linewidth=0.5,
          alpha=0.5,
          # label=f"{follower.id} Trajectory",
        )
      
      ax_env.set_xlim(0, self.ENV_WIDTH)
      ax_env.set_ylim(0, self.ENV_HEIGHT)
      ax_env.set_title('Explore Environment')
      ax_env.set_xlabel('X')
      ax_env.set_ylabel('Y')
      ax_env.grid(False)
      ax_env.legend()

      # ---------------------------------------- Pdf ----------------------------------------
      # pdfのグリッドを設定
      gs_pdf = gridspec.GridSpecFromSubplotSpec(
        nrows=2,
        ncols=3,
        subplot_spec=gs_master[1:, :],
      )

      # PDFのグリッドを6分割
      pdf_axes: list[plt.Axes] = []
      titles = [
        "drivability (Cartesian plot)",
        "exploration_improvement (Cartesian plot)",
        "combined (Cartesian plot)",
        "drivability (polar plot)",
        "exploration_improvement (polar plot)",
        "combined (polar plot)"
      ]
      for i in range(2):
        for j in range(3):
            if i == 1:  # 2段目（下段）は極座標
                ax = fig.add_subplot(gs_pdf[i, j], projection='polar')
            else:  # 1段目（上段）は直交座標
                ax = fig.add_subplot(gs_pdf[i, j])
            pdf_axes.append(ax)
      
      # 走行可能性の確率分布の描画(cartesian)
      pdf_axes[0].set_title(titles[0])
      
      for data in self.drivability_pdf_list:
        pdf_axes[0].plot(
          self.ANGLES,
          data['pdf'],
          label=data['id'],
          linestyle=data['lineStyle'],
          linewidth=data['lineWidth'],
        )
      pdf_axes[0].set_xlabel("Angle(radians)")
      pdf_axes[0].set_ylabel("Probability")
      pdf_axes[0].legend()

      # 探査向上性の確率分布の描画(cartesian)
      pdf_axes[1].set_title(titles[1])
      for data in self.explore_improvement_pdf_list:
        pdf_axes[1].plot(
          self.ANGLES,
          data['pdf'],
          label=data['id'],
          linestyle=data['lineStyle'],
          linewidth=data['lineWidth'],
        )
      pdf_axes[1].set_xlabel("Angle(radians)")
      pdf_axes[1].set_ylabel("Probability")
      pdf_axes[1].legend()

      # 結合確率分布の描画(cartesian)
      pdf_axes[2].set_title(titles[2])
      for data in self.result_pdf_list:
        pdf_axes[2].plot(
          self.ANGLES,
          data['pdf'],
          label=data['id'],
          linestyle=data['lineStyle'],
          linewidth=data['lineWidth'],
        )
      pdf_axes[2].set_xlabel("Angle(radians)")
      pdf_axes[2].set_ylabel("Probability")
      pdf_axes[2].legend()

      # 走行可能性の確率分布の描画(polar)
      pdf_axes[3].set_title(titles[3])
      for data in self.drivability_pdf_list:
        pdf_axes[3].plot(
          self.ANGLES,
          data['pdf'],
          label=data['id'],
          linestyle=data['lineStyle'],
          linewidth=data['lineWidth'],
        )
      pdf_axes[3].legend()

      # 探査向上性の確率分布の描画(polar)
      pdf_axes[4].set_title(titles[4])
      for data in self.explore_improvement_pdf_list:
        pdf_axes[4].plot(
          self.ANGLES,
          data['pdf'],
          label=data['id'],
          linestyle=data['lineStyle'],
          linewidth=data['lineWidth'],
        )
      pdf_axes[4].legend()

      # 結合確率分布の描画(polar)
      pdf_axes[5].set_title(titles[5])
      for data in self.result_pdf_list:
        pdf_axes[5].plot(
          self.ANGLES,
          data['pdf'],
          label=data['id'],
          linestyle=data['lineStyle'],
          linewidth=data['lineWidth'],
        )
      pdf_axes[5].legend()

      # レイアウトの調整
      plt.tight_layout()  
      # plt.draw()
      # plt.pause(0.001) # TODO 一旦出力なくす

      # フレームの保存
      if save_frames:
        filename = f"frame_{len(self.env_frames)}.png"
        self.env_frames.append(fig)
        plt.savefig(filename) 
  

  def save_gif(self, episode, date_time) -> None:
    """
    保存したフレームをGIFに変換
    """
    path_name = f"{date_time}_episode{episode}"
    gif_dir = f"gif/{path_name}"
    gif_name = f"{path_name}.gif"

    os.makedirs(gif_dir, exist_ok=True)

    for i, frame in enumerate(self.env_frames):
      frame_path = f'frame_{i}.png'
      frame.savefig(frame_path)
      plt.close(frame)

    # images = [imageio.imread(frame) for frame in self.env_frames]
    images = [imageio.imread(f'frame_{i}.png') for i in range(len(self.env_frames))]
    imageio.mimsave(f"{gif_dir}/{gif_name}", images, duration=0.1)
    


    # 一時ファイルの削除
    for i in range(len(self.env_frames)):
      os.remove(f"frame_{i}.png")
    
    self.env_frames = []


  def step(self, action):
    """
    action:
      parameter B of the policy (continuity)    | (0 <= B <= 1): float
      parameter k_d of the policy (continuity)  | (0 <= k_d < inf): float
      parameter k_e of the policy (continuity)  | (0 <= k_e < inf): float
      parameter k_c of the policy (continuity)  | (0 <= k_c < inf): float
    """
    # actionの取得
    B = action["B"]
    k_d = action["k_d"]
    k_e = action["k_e"]
    k_c = action["k_c"]

    print(f"B : {B}, k_d: {k_d}, k_e; {k_e}, k_c: {k_c}")
    
    self.result_pdf_list = [] # 結合確率分布の初期化
    # 走行可能性の確率分布の生成, フォロワーの衝突情報の取得
    drivability, follower_collision_info_list = self.get_drivability(k_d)
    self.result_pdf_list.append({
      "id": "drivability",
      "pdf": drivability,
      "lineStyle": "--",
      "lineWidth": 1
    })

    print(f"drivability :{drivability}")
    
    # 探査向上性の確率分布の生成
    exploration_improvement = self.get_exploration_improvement(k_e, k_c)
    self.result_pdf_list.append({
      "id": "exploration_improvement",
      "pdf": exploration_improvement,
      "lineStyle": "--",
      "lineWidth": 1
    })

    print(f"exploration_improvement :{exploration_improvement}")
    
    # 最終的な確率分布を生成
    output_pdf = B * drivability + (1 - B) * exploration_improvement
    print(f"before output_pdf : {output_pdf}")
    output_pdf = output_pdf / np.sum(output_pdf)
    
    self.result_pdf_list.append({
      "id": f"combined(B={B})",
      "pdf": output_pdf,
      "lineStyle": "-",
      "lineWidth": 2
    })

    print(f"output_pdf : {output_pdf}")

    # 確率分布から次の移動方向を決定
    theta = np.random.choice(self.ANGLES, p=output_pdf)
    print(f"theta: {theta}")

    dx = self.OUTER_BOUNDARY * np.cos(theta)
    dy = self.OUTER_BOUNDARY * np.sin(theta)

    self.previous_agent_position = self.agent_position.copy()
    self.agent_position, collision_flag = self.next_position(dy, dx)
    self.agent_trajectory.append(self.agent_position.copy())

    # フォロワーのリーダー機の観測位置の変更
    for index in range(len(self.follower_robots)):
      self.follower_robots[index].change_agent_state(self.agent_position)

    # leader_collision_pointの獲得
    if collision_flag:
      leader_collision_point = {
        "value": theta,
        "is_none": 0
      }
    else:
      leader_collision_point = {
        "value": theta,
        "is_none": 1
      }

    # フォロワーの探査行動
    for _ in range(self.FOLLOWER_STEP):
      for index in range(len(self.follower_robots)):
        previous_position = self.follower_robots[index].point
        self.follower_robots[index].step_motion()
        self.update_exploration_map(previous_position, self.follower_robots[index].point) # 探査状況の更新

      # レンダリング
      self.render(save_frames=self.SAVE_FRAMES, mode='human')
    
    # 報酬計算
    reward = self.REWARD_DEFAULT
    if collision_flag:
      # リーダーが障害物に衝突した場合
      reward += self.REWARD_LEADER_COLLISION
    else:
      # 探査率の計算
      exploration_rate = self.explored_area / self.total_area
      if exploration_rate > self.previous_explored_area:
        # 探査率が上昇した場合 # TODO 上昇した量により報酬を変更するようにする
        reward += self.REWARD_EXPLORATION
    
    # フォロワーの報酬計算
    for collision_info in follower_collision_info_list:
      reward -= collision_info['count'] * self.REWARD_FOLLOWER_COLLISION

    # 終了条件
    done = False
    turncated = False # TODO エピソードが途中で終了した場合
    if self.explored_area >= self.total_area * self.FINISH_EXPLORED_RATE: # 探査率が一定以上になった場合
      done = True
      reward += self.REWARD_FINISH

    self.state = {
      "follower_collision_info_list": follower_collision_info_list,
      "leader_collision_point": leader_collision_point,
      "B": B,
      "k_d": k_d,
      "k_e": k_e,
      "k_c": k_c,
    }

    return self.state, reward, done, turncated, {} # TODO infoを返す必要があるかも


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
          stop_position = self.agent_position + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)

          return stop_position, collision_flag
      
      else:
        continue

    return self.agent_position + np.array([dy, dx]), collision_flag
  

  def get_drivability(self, k_d) -> np.ndarray:
    """
    走行可能性の確率分布の生成
    """
    self.drivability_pdf_list = []
    follower_collision_info_list = []
    combined_pdf = np.zeros_like(self.ANGLES)

    for follower in self.follower_robots:
      data = follower.calculate_collision_stats() # red_id, count, mean, variance
      collision_info = {
        "has_collisions": data['has_collisions'],
        "mean": data['mean'],
        "covariance": data['covariance'],
        "count": data['count'],
      }
      follower_collision_info_list.append(collision_info)

      # azimuth = self.calculate_follower_azimuth(data['mean'])

      if data['has_collisions'] == 1:
        count = data['count']
        covariance = data['covariance']
        kappa = self.calculate_drivability_kappa(count, covariance, k_d) # 逆温度の計算

        print(f"count: {count} | covariance: {covariance} | kappa: {kappa} | mean: {data['mean']}")


        follower_pdf = vonmises.pdf(self.ANGLES, kappa, loc=data['mean']) # フォロワーの確率分布
        print(f"red_id {follower.id} | red_data: {data}")
        print(f"red_id: {follower.id} | follower_pdf :{follower_pdf}")

        follower_pdf /= np.sum(follower_pdf) # 正規化

        # 分布を反転
        follower_pdf = 1 - follower_pdf
        follower_pdf /= np.sum(follower_pdf) # 正規化

        self.drivability_pdf_list.append({
          "id": data['red_id'],
          "pdf": follower_pdf,
          "lineStyle": "--",
          "lineWidth": 1
        })
        combined_pdf += follower_pdf
    
    print(f"combined_pdf: {combined_pdf.sum()}")

    if combined_pdf.sum() > 0:
      combined_pdf /= np.sum(combined_pdf)
    else:
      # 衝突データが存在しない場合
      print("No collision data. using uniform distribution.")
      combined_pdf = np.ones_like(self.ANGLES) # 一様分布を出力

    self.drivability_pdf_list.append({
      "id": "combined",
      "pdf": combined_pdf,
      "lineStyle": "-",
      "lineWidth": 2
    })

    return combined_pdf, follower_collision_info_list
  

  def get_exploration_improvement(self, k_e, k_c) -> np.ndarray:
    """
    探査向上性の確率分布の生成
    """
    self.explore_improvement_pdf_list = []
    mu = self.calculate_previous_azimuth()
    combined_pdf = np.zeros_like(self.ANGLES)

    if mu is not None:
      previous_state_pdf = vonmises.pdf(self.ANGLES, k_e, loc=mu) # 前回の状態から得られる確率分布
      previous_state_pdf /= np.sum(previous_state_pdf)
      combined_pdf += previous_state_pdf
      self.explore_improvement_pdf_list.append({
        "id": "previous_state",
        "pdf": previous_state_pdf,
        "lineStyle": "--",
        "lineWidth": 1
      })
    else:
      print("No previous state.")
    
    
    # 衝突があった場合, 衝突方向の確率分布を下げる
    if self.state['leader_collision_point']['is_none']:
      if mu is not None: # TODO muの計算変更
        collision_pdf = vonmises.pdf(self.ANGLES, k_c, loc=mu) 
        collision_pdf /= np.sum(collision_pdf)

        collision_pdf = 1 - collision_pdf
        collision_pdf /= np.sum(collision_pdf)

        combined_pdf += collision_pdf
        self.explore_improvement_pdf_list.append({
          "id": "collision_pdf",
          "pdf": collision_pdf,
          "lineStyle": "--",
          "lineWidth": 1
        })
    else:
      print("No collision state") 

    
    self.explore_improvement_pdf_list.append({
      "id": "combined",
      "pdf": combined_pdf,
      "lineStyle": "-",
      "lineWidth": 2
    })
    
    return combined_pdf


  def calculate_follower_azimuth(self, follower_position: np.ndarray) -> float:
    """
    リーダー機から見たフォロワー機の方位角を計算
    障害物情報の中心となる方位角を計算
    """
    # print(f"follower_position: {follower_position}")

    if follower_position is not None:
      # 現在位置と前の位置の差分を計算
      dy = follower_position[0] - self.agent_position[0] 
      dx = follower_position[1] - self.agent_position[1]

      # atan2で方位角を計算
      azimuth = math.atan2(dy, dx)

      # 0~2πに正規化
      if azimuth < 0:
        azimuth += 2 * math.pi
    else:
      azimuth = None

    return azimuth 
  

  def calculate_drivability_kappa(self, count, covariance, k_d):
    """
    逆温度の計算
    TODO 改良の余地あり
    """
    if covariance == 0:
      return 1.0 # 分散が0の場合は1.0
    
    return k_d * (count / covariance)
  

  def calculate_previous_azimuth(self) -> float:
    """
    前回の方位角を計算
    """
    if self.previous_agent_position is not None:
      # 現在位置と前の位置の差分を計算
      dy = self.agent_position[0] - self.previous_agent_position[0]
      dx = self.agent_position[1] - self.previous_agent_position[1]

      # atan2で方位角を計算
      azimuth = math.atan2(dy, dx)

      # 0~2πに正規化
      if azimuth < 0:
        azimuth += 2 * math.pi
    else:
      azimuth = None
    
    return azimuth
  

  def calculate_collision_azimuth(self) -> float:
    """
    衝突が発生した場合の方位角を計算
    """
    dy = self.previous_agent_position[0] - self.agent_position[0]
    dx = self.previous_agent_position[1] - self.agent_position[1]

    azimuth = math.atan(dy, dx)

    if azimuth < 0:
      azimuth += 2 * math.pi
    
    return azimuth
  

  def update_exploration_map(self, previous_position, current_position) -> None:
    """
    探査状況の更新
    """
    # 前回位置から現在位置までの線分を取得
    line_points = self.interpolate_line(previous_position, current_position)

    for y, x in line_points:
      if 0 <= y < self.ENV_HEIGHT and 0 <= x < self.ENV_WIDTH:
        if self.explored_map[y, x] == 0 and self.map[y, x] != self.OBSTACLE_VALUE:
          self.explored_map[y, x] = 1
          self.explored_area += 1
  

  def interpolate_line(self, p1, p2):
    """
    2点間の整数座標の線分を取得(Bresenhamアルゴリズムの応用)
    :p1: 始点(y, x)
    :p2: 終点(y, x)
    return: 線分上の全ての整数座標
    """
    y1, x1 = int(p1[0]), int(p1[1])
    y2, x2 = int(p2[0]), int(p2[1])

    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((y1, x1))
        if y1 == y2 and x1 == x2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points    

  
  def close(self):
    """
    環境を終了する際のクリーンアップ処理
    """
    # Matplotlib のプロットウィンドウを閉じる
    plt.close('all')


  def seed(self, seed=None):
    """
    環境のランダムシードを設定
    """
    self.np_random, seed = gym.utils.seeding.np_random(seed)
    random.seed(seed)
    np.random.seed(seed)
    return [seed]
  

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
  

  