# from envs.group_behavior.group_behavior_strategy_env import GroupBehaviorStrategyEnv

import pandas  as pd
import numpy as np
import random
import math
import os

class Red():
    """
    REDの確率密度制御を模倣したクラス
    x                   : x座標
    y                   : y座標
    point               : 座標
    amount_of_movement  : 移動量
    direction_angle     : ロボット正面からの移動角度
    """
    MIN_MOVEMENT            = 2.0   # 最小直進量
    MAX_MOVEMENT            = 3.0   # 最大直進量
    MAX_BOIDS_MOVEMENT      = 3.0   # boids最大直進量
    MIN_BOIDS_MOVEMENT      = 2.0   # boids最小直進量
    AVOIDANCE_BEHAVIOR_MIN  = 90.0  # 回避行動最小角度
    AVOIDANCE_BEHAVIOR_MAX  = 270.0  # 回避行動最大角度

    def __init__(
            self,
            id: str,
            env,
            agent_position: np.array,
            x: float,
            y: float,
            step: int = 0, 
            amount_of_movement: float = 0.0, 
            direction_angle: float = 0.0, 
            collision_flag: bool = False, 
            boids_flag: int = 0, 
            estimated_probability: float = 0.0, 
            ):
        """
        REDのコンストラクタ
        """
        self.id: str                        = id
        self.env                            = env
        self.agent_position: np.array       = agent_position
        self.step: int                      = step
        self.x: float                       = x
        self.y: float                       = y
        self.point: np.array                = np.array([self.y, self.x])
        self.amount_of_movement: float      = amount_of_movement
        self.direction_angle: float         = direction_angle
        self.distance: float                = np.linalg.norm(self.point - agent_position)
        self.azimuth: float                 = self.azimuth_adjustment()
        self.collision_flag: bool           = collision_flag
        self.boids_flag: int                = boids_flag
        self.estimated_probability: float   = estimated_probability
        self.data = self.get_arguments()
        self.one_explore_data = self.get_arguments()
    

    def get_arguments(self):
        """
        データをデータフレーム形式にする
        """
        return pd.DataFrame({'id': [self.id],
                             'step': [self.step], 
                             'agent_position': [self.agent_position],
                             'x': [self.x], 
                             'y': [self.y], 
                             'point': [self.point], 
                             'amount_of_movement': [self.amount_of_movement], 
                             'direction_angle': [self.direction_angle], 
                             'distance': [self.distance], 
                             'azimuth': [self.azimuth],
                             'collision_flag': [self.collision_flag],
                             'boids_flag': [self.boids_flag],
                             'estimated_probability': [self.estimated_probability],
                             })
    

    def save_to_db(self, engine):
        """
        データベースに保存
        """
        self.data.to_sql('red_agents', engine, if_exists='append', index=False)
    

    def get_csv(self, episode, date_time):
        """
        データをcsv形式で保存
        """
        directory = f"csv/{date_time}_episode{episode}"
        os.makedirs(directory, exist_ok=True)
        self.data.to_csv(f"{directory}/{self.id}.csv")


    def azimuth_adjustment(self) -> float:
        """
        探査中心の方向角を計算
        """
        azimuth: float = 0.0
        if self.x != self.agent_position[1]:
            vec_d = np.array(self.point - self.agent_position)
            vec_x = np.array([0, self.x - self.agent_position[1]])

            azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
        
        if self.x - self.agent_position[1] < 0:
            if self.y - self.agent_position[0] >= 0:
                azimuth = np.rad2deg(math.pi) - azimuth
            else:
                azimuth += np.rad2deg(math.pi)
        
        else:
            if self.y - self.agent_position[0] < 0:
                azimuth = np.rad2deg(2.0 * math.pi) - azimuth

        # # ロボットからエージェントへのベクトルを計算
        # dy = self.agent_position[0] - self.y  # y方向の差
        # dx = self.agent_position[1] - self.x  # x方向の差

        # # 偏角を計算 (度単位)
        # azimuth = np.degrees(np.arctan2(dy, dx))

        # # 正規化して0~360度に変換
        # azimuth = (azimuth + 360) % 360
        
        return azimuth


    def avoidance_behavior(self) -> np.array:
        """
        障害物回避行動
        """
        self.direction_angle = (self.direction_angle + random.uniform(self.AVOIDANCE_BEHAVIOR_MIN, self.AVOIDANCE_BEHAVIOR_MAX)) % np.rad2deg(math.pi * 2.0)
        amount_of_movement = random.uniform(self.MIN_MOVEMENT, self.MAX_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.y + dy, self.x + dx])
        return prediction_point
    

    def forward_behavior(self, dy, dx) -> np.array:
        """
        直進行動処理
        """
        SAMPLING_NUM = max(150, int(np.ceil(np.linalg.norm([dy, dx]) * 10)))
        SAFE_DISTANCE = 1.0 # マップの安全距離

        for i in range(1, SAMPLING_NUM + 1):
            intermediate_position = np.array([
                self.point[0] + (dy * i / SAMPLING_NUM),
                self.point[1] + (dx * i / SAMPLING_NUM)
            ])

            if (0 < intermediate_position[0] < self.env.ENV_HEIGHT) and (0 < intermediate_position[1] < self.env.ENV_WIDTH):
                map_y = int(intermediate_position[0])
                map_x = int(intermediate_position[1])

                if self.env.map[map_y, map_x] == self.env.OBSTACLE_VALUE:
                    # 障害物に衝突する事前位置を計算
                    collision_position = intermediate_position
                    direction_vector = collision_position - self.point
                    norm_direction_vector = np.linalg.norm(direction_vector)

                    stop_position = self.point + (direction_vector / norm_direction_vector) * (norm_direction_vector - SAFE_DISTANCE)

                    self.collision_flag = True

                    return stop_position
            else:
                continue
        
        self.collision_flag = False

        return self.point + np.array([dy, dx])
    

    def boids_judgement(self) -> None:
        """
        boids行動を行うか判断する
        """
        self.distance = np.linalg.norm(self.point - self.agent_position)
        if self.distance > self.env.OUTER_BOUNDARY:
            self.boids_flag = 1
        elif self.distance < self.env.INNER_BOUNDARY:
            self.boids_flag = 2
        else:
            self.boids_flag = 0
    

    def boids_behavior(self) -> np.array:
        """
        boids行動
        """
        self.direction_angle = self.azimuth
        if self.boids_flag == 1:
            if self.y - float(self.agent_position[0]) >= 0:
                self.direction_angle += np.rad2deg(math.pi)
            else:
                self.direction_angle -= np.rad2deg(math.pi)
        
        amount_of_movement = random.uniform(self.MIN_BOIDS_MOVEMENT, self.MAX_BOIDS_MOVEMENT)
        dx = amount_of_movement * math.cos(math.radians(self.direction_angle))
        dy = amount_of_movement * math.sin(math.radians(self.direction_angle))
        prediction_point = np.array([self.y + dy, self.x + dx])
        return prediction_point
    

    def rejection_decision(self) -> np.array:
        """
        メトロポリス法による棄却決定
        """
        def distribution(distance, mean, variance) -> float:
            """
            正規分布
            """
            return 1 / math.sqrt(2 * math.pi) * math.exp(-(distance - mean) ** 2 / (2 * variance ** 2))
        

        while True:
            direction_angle = np.rad2deg(random.uniform(0.0, 2.0 * math.pi))
            amount_of_movement = random.uniform(self.MIN_MOVEMENT, self.MAX_MOVEMENT)
            dx = amount_of_movement * math.cos(math.radians(direction_angle))
            dy = amount_of_movement * math.sin(math.radians(direction_angle))
            prediction_point = np.array([self.y + dy, self.x + dx])
            distance = np.linalg.norm(prediction_point - self.agent_position)
            estimated_probability = distribution(distance, self.env.MEAN, self.env.VARIANCE)
            if self.estimated_probability == 0.0:
                self.estimated_probability = estimated_probability
                self.direction_angle = direction_angle
                return prediction_point
            else:
                if estimated_probability / self.estimated_probability > np.random.rand():
                    self.estimated_probability = estimated_probability
                    self.direction_angle = direction_angle
                    return prediction_point
                else:
                    continue

    
    def step_motion(self) -> None:
        """
        行動制御
        """
        if self.collision_flag:
            prediction_point = self.avoidance_behavior()
        else:
            self.boids_judgement()
            if self.boids_flag:
                prediction_point = self.boids_behavior()
            else:
                prediction_point = self.rejection_decision()
        
        self.point = self.forward_behavior(
            prediction_point[0] - self.point[0],
            prediction_point[1] - self.point[1]
            )
        
        self.y = self.point[0]
        self.x = self.point[1]
        self.distance = np.linalg.norm(self.point - self.agent_position)
        self.azimuth = self.azimuth_adjustment()
        self.step += 1

        self.data = pd.concat([self.data, self.get_arguments()])
        self.one_explore_data = pd.concat([self.one_explore_data, self.get_arguments()])
    

    def change_agent_state(self, agent_position) -> None:
        """
        エージェントの状態が変化した場合
        """
        self.agent_position = agent_position
        self.one_explore_data = self.get_arguments()
    

    def calculate_collision_stats(self) -> dict:
        """
        衝突統計量の計算
        Returns
            collision_stats: dict
              red_id: str
              count: int
              mean: float
              cov: float
        """
        collision_data = self.data[self.data['collision_flag'] == True]

        # print(f"red: {self.id} | collision_data: {collision_data}")

        collision_stats = {
            'red_id': self.id,
            'has_collisions': 0,
            'count': 0,
            'mean': 0.0,
            'covariance': 0.0
        }

        if not collision_data.empty:
            mean_y = collision_data['y'].mean()
            mean_x = collision_data['x'].mean()

            # covariance = collision_data[['y', 'x']].cov().values
            # 要素が2つ以上ある場合は共分散を計算
            if len(collision_data) > 1:
                covariance = abs(collision_data['y'].cov(collision_data['x'])) # 絶対値を使用
            else:
                covariance = 0

            azimuth: float = 0.0
            if mean_x != self.agent_position[1]:
                vec_d = np.array([mean_y - self.agent_position[0], mean_x - self.agent_position[1]])
                vec_x = np.array([0, mean_x - self.agent_position[1]])

                azimuth = np.rad2deg(math.acos(vec_d @ vec_x / (np.linalg.norm(vec_d) * np.linalg.norm(vec_x))))
            
            if mean_x - self.agent_position[1] < 0:
                if mean_y - self.agent_position[0] >= 0:
                    azimuth = np.rad2deg(math.pi) - azimuth
                else:
                    azimuth += np.rad2deg(math.pi)
            
            else:
                if mean_y - self.agent_position[0] < 0:
                    azimuth = np.rad2deg(2.0 * math.pi) - azimuth
        
            collision_stats = {
                'red_id': self.id,
                'has_collisions': 1,
                'count': collision_data.shape[0],
                'mean': azimuth,
                'covariance': covariance
            }
        
        # print(f"collision_stats: {collision_stats}")

        return collision_stats
        

    def __str__(self):
        return f'red[{self.id}], step:{self.step},position:{self.point}'
