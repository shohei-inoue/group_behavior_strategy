import tensorflow as tf
from models.actor_critic_model3 import Actor, Critic

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.gamma = gamma  # 割引率

    def select_action(self, state):
        """Actor を使って行動を選択"""
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        action_probs = self.actor(state)
        action = action_probs.numpy()[0]  # [0, 1] 範囲の連続値
        return action

    def update(self, state, action, reward, next_state, done):
        """Actor-Critic の更新"""
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape_critic:
            # Critic の損失計算
            value = self.critic(tf.expand_dims(state, axis=0))
            next_value = self.critic(tf.expand_dims(next_state, axis=0))
            target = reward + self.gamma * next_value * (1 - done)
            critic_loss = tf.math.reduce_mean(tf.square(target - value))

        critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape_actor:
            # Actor の損失計算
            value = self.critic(tf.expand_dims(state, axis=0))
            next_value = self.critic(tf.expand_dims(next_state, axis=0))
            target = reward + self.gamma * next_value * (1 - done)
            advantage = target - value

            # アクションの選択に基づく損失を計算
            action_probs = self.actor(tf.expand_dims(state, axis=0))
            log_prob = tf.math.log(action_probs + 1e-8)  # 安定性のための 1e-8
            actor_loss = -tf.reduce_mean(log_prob * advantage)

        actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
