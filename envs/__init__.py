from gym.envs.registration import register

register(
    id='group_behavior_strategy-v0',
    entry_point='envs.group_behavior.group_behavior_strategy_env:GroupBehaviorStrategyEnv',
)