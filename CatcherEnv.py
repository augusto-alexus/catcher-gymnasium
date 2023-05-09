import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

class CatcherEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}


    def __init__(self, render_mode=None, window_size=128, agent_lives=3, catch_reward=1, miss_reward=-1, lose_reward=-5):
        self.window_size = window_size

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, window_size, shape=(1,), dtype=np.int16),
                "target": spaces.Box(0, window_size, shape=(2,), dtype=np.int16),
            }
        )

        self.catch_reward = catch_reward
        self.miss_reward = miss_reward
        self.lose_reward = lose_reward

        self._agent_lives = agent_lives
        self._agent_lives_left = agent_lives
        self._agent_velocity = int(window_size * 0.06)
        self._target_velocity = int(window_size * 0.03)

        self._episodes = 0
        self._episode_reward = 0
        self.rewards = []

        self.action_space = spaces.Discrete(2)
        self._action_to_direction = {
            0: np.array([-1, 0]),
            1: np.array([1, 0]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([int(0.5 * self.window_size), int(0.95 * self.window_size)])
        self._agent_lives_left = self._agent_lives

        self._reset_target_location()

        if self._episodes > 0:
            self.rewards.append(self._episode_reward)
        self._episode_reward = 0
        self._episodes += 1

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    def step(self, action: int):
        agent_movement_direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + agent_movement_direction * self._agent_velocity, 0, self.window_size)
        self._target_location = np.clip(self._target_location + np.array([0, 1]) * self._target_velocity, 0, self.window_size)

        reward = 0
        missed = self._target_location[1] == self.window_size
        if missed:
            self._agent_lives_left -= 1
            reward = self.miss_reward
        catched = pygame.Rect.colliderect(self._get_agent_rect(), self._get_target_rect())
        if catched:
            reward = self.catch_reward
        if catched or missed:
            self._reset_target_location()
        terminated = 1 if self._agent_lives_left == 0 else 0
        if terminated:
            reward = self.lose_reward
        self._episode_reward += reward
        
        info = self._get_info()
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info


    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _reset_target_location(self):
        x_rand = self.np_random.random() * 0.9 + 0.05 # within [0.05; 0.95]
        self._target_location = np.array([int(x_rand * self.window_size), 0])

    def _get_agent_rect(self):
        return pygame.Rect(self._agent_location[0], self._agent_location[1], int(self.window_size * 0.2), int(self.window_size * 0.04))
    
    def _get_target_rect(self):
        return pygame.Rect(self._target_location[0], self._target_location[1], int(self.window_size * 0.06), int(self.window_size * 0.06))

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((25, 25, 25))

        pygame.draw.rect(canvas, (255, 0, 0), self._get_target_rect())
        pygame.draw.rect(canvas, (255, 255, 255), self._get_agent_rect())

        assert self.window is not None
        assert self.clock is not None

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


    def _get_obs(self):
        return {"agent": self._agent_location[0], "target": self._target_location}
    
    def _get_info(self):
        return {}