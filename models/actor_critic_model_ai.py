import tensorflow as tf
from tensorflow.keras import layers

class ActorCriticModel(tf.keras.Model):
    def __init__(self, hidden_size=128):
        """
        Actor-Critic Model with shared layers for feature extraction.
        
        Args:
            hidden_size (int): Number of units in hidden layers.
        """
        super(ActorCriticModel, self).__init__()

        # Shared network for feature extraction
        self.shared = tf.keras.Sequential([
            layers.Flatten(),  # Flatten the input
            layers.Dense(hidden_size, activation='relu'),
            layers.Dense(hidden_size, activation='relu')
        ])

        # Actor network for each action
        self.actor_B = layers.Dense(1, activation='sigmoid')  # 0 <= B <= 1
        self.actor_k_d = layers.Dense(1, activation='softplus')  # 0 <= k_d < inf
        self.actor_k_e = layers.Dense(1, activation='softplus')  # 0 <= k_e < inf
        self.actor_k_c = layers.Dense(1, activation='softplus')  # 0 <= k_c < inf

        # Critic network
        self.critic = layers.Dense(1)  # State value

    def call(self, inputs):
        """
        Forward pass for the model.

        Args:
            inputs (dict): A dictionary containing observation space inputs.

        Returns:
            actions (dict): Predicted actions for each component.
            value (tf.Tensor): Predicted state value.
        """
        combined_inputs = tf.concat([
          inputs['follower_collision_info_list'],  # 統合されたフォロワー情報 [4]
          inputs['leader_collision_point'],  # 形状: [1, 2]
          tf.reshape(inputs['B'], [-1, 1]),  # 形状: [1, 1]
          tf.reshape(inputs['k_d'], [-1, 1]),  # 形状: [1, 1]
          tf.reshape(inputs['k_e'], [-1, 1]),  # 形状: [1, 1]
          tf.reshape(inputs['k_c'], [-1, 1])  # 形状: [1, 1]
        ], axis=-1)


        # Pass through the shared network
        features = self.shared(combined_inputs)

        # Generate actions from actor network
        actions = {
            'B': self.actor_B(features),
            'k_d': self.actor_k_d(features),
            'k_e': self.actor_k_e(features),
            'k_c': self.actor_k_c(features)
        }

        # Generate value from critic network
        value = self.critic(features)

        return actions, value
