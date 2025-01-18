import tensorflow as tf
from tensorflow.keras import layers

class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc_out = layers.Dense(action_dim, activation='sigmoid')  # アクション出力 (0~1 範囲)

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        action = self.fc_out(x)
        return action

class Critic(tf.keras.Model):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(128, activation='relu')
        self.fc_out = layers.Dense(1)  # 状態価値のスカラー出力

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        value = self.fc_out(x)
        return value
