import gymnasium.spaces
import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled

# rendering colors
BLACK = (0, 0, 0)            # unexplored cell
WHITE = (255, 255, 255)      # explored cell
BROWN = (101, 67, 33)        # wall
GREY = (160, 161, 161)       # agent

# color IDs
COLOR_IDS = {
    0: BLACK,      # unexplored cell
    1: WHITE,      # explored cell
    2: BROWN,      # wall
    3: GREY        # agent
}

# action IDs
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4


class SimpleGridworld(gym.Env):
    """
    Gridworld where the agent has to explore all tiles while avoiding obstacles.
    """
    metadata = {
        "render_modes": ["human"],
        "render_fps": 10,
        "grid_size": 3
    }

    def __init__(self, render_mode="human"):
        # Grid attributes
        self.grid_size = self.metadata["grid_size"]
        self.blueprint_grid = [[3, 0, 0],
                                [0, 2, 0],
                                [0, 0, 0]]
        self.grid = np.asarray(grid, dtype=np.uint8)

        # Gymnasium spaces
        grid_space = np.zeros((3, 3)) + 4
        self.observation_space = gym.spaces.MultiDiscrete(grid_space)   # TODO: define observation space
        self.action_space = gymnasium.spaces.Discrete(5)   # TODO: define action space

        # Rendering attributes for Pygame
        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.window_size = (
            min(64 * self.grid_size, 512),
            min(64 * self.grid_size, 512)
        )
        self.tile_size = (
            self.window_size[0] // self.grid_size,
            self.window_size[1] // self.grid_size,
        )

        # State attributes
        self.agent_pos = 0   # agent position, considering the flattened grid (e.g. cell 2,2 is position 8)
        self.total_covered_cells = 1   # how many cells have been covered by the agent so far
        self.coverable_cells = 8   # how many cells can be covered in the current map layout
        self.steps_remaining = 500   # steps remaining in the episode
        self.game_over = False   # if the episode has ended or not

    def reset(self, **kwargs):
        """
        Required Gymnasium method, resets the environment for a new episode of training
        """
        super().reset(**kwargs)

        # TODO: Fill in reset method
        # ...
        self.agent_pos = 0  # agent position, considering the flattened grid (e.g. cell 2,2 is position 8)
        self.total_covered_cells = 1  # how many cells have been covered by the agent so far
        self.steps_remaining = 500  # steps remaining in the episode
        self.grid = np.asarray(self.blueprint_grid, dtype=np.uint8)

        # Renders map, if render_mode is "human"
        if self.render_mode is not None and self.render_mode == "human":
            self.render()

        return None, {}

    def step(self, action: int):
        """
        Required Gymansium method, performs a step within the environment given the action provided
        """
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = None

        # TODO: Fill in step() method
        # ...

        new_cell_covered = self.move(action)

        return observation, reward, terminated, truncated, info

    def move(self, action):
        """
        Moves the agent within the grid based on the action provided. Returns True if a new cell is covered
        """
        new_cell_covered = False

        # TODO: Complete move() method updating grid variables
        # ...

        return new_cell_covered

    def render(self):
        """
        Renders grid to a Pygame window
        """
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption(self.unwrapped.spec.id)
            self.window_surface = pygame.display.set_mode(self.window_size)

        assert (
                self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."

        if self.clock is None:
            self.clock = pygame.time.Clock()

        t_size = self.tile_size  # short notation

        # draw tiles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                pos = (x * t_size[0], y * t_size[1])
                border = pygame.Rect(pos, tuple(cs * 1.01 for cs in t_size))
                rect = pygame.Rect(pos, tuple(cs * 0.99 for cs in t_size))

                # draw background
                if self.grid[y, x] == 1:  # draws black border if cell is white
                    pygame.draw.rect(self.window_surface, BLACK, border)
                else:
                    pygame.draw.rect(self.window_surface, WHITE, border)

                if y * self.grid_size + x == self.agent_pos:  # draw agent's cell
                    pygame.draw.rect(self.window_surface, WHITE, rect)
                    agent_color = GREY
                    pygame.draw.ellipse(self.window_surface, agent_color, rect)
                else:  # draw other cells
                    cell_id = int(self.grid[y, x])
                    cell_color = COLOR_IDS[cell_id]
                    pygame.draw.rect(self.window_surface, cell_color, rect)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            raise NotImplementedError

    def close(self):
        """
        Closes Pygame's window
        """
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
