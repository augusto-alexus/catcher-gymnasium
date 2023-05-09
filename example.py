import gymnasium as gym
from gymnasium.envs.registration import register

register(
     id="CatcherEnv-v0",
     entry_point="CatcherEnv:CatcherEnv",
     max_episode_steps=None,
)

env_demo = gym.make("CatcherEnv-v0", render_mode="human")
observation, info = env_demo.reset()

done = False
while not done:
    action = env_demo.action_space.sample()
    observation, reward, terminated, truncated, info = env_demo.step(action)

    if terminated or truncated:
        done = True

env_demo.close()