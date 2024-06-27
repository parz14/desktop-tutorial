import sys
import pygame as pg
import gymnasium as gym
import numpy as np

class ChidEnv(gym.Env):
    def __init__(self, goal_coordinates=(4, 4), hell_state_coordinates=None, grid_size=5) -> None:
        super(ChidEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 100
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array(goal_coordinates)
        self.done = False
        self.hell_states = [np.array(hell) for hell in (hell_state_coordinates or [])]
        self.hell_hit_count = 0

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize Pygame and the window:
        pg.init()
        self.screen = pg.display.set_mode((self.cell_size * self.grid_size, self.cell_size * self.grid_size))
        pg.display.set_caption("ChidEnv Game")

        # Load the galaxy background image
        self.background_image = pg.image.load('pexels-photo-2098427.jpeg')
        self.background_image = pg.transform.scale(self.background_image, (self.cell_size * self.grid_size, self.cell_size * self.grid_size))
        
        # Load the UFO image
        self.ufo_image = pg.image.load('istockphoto-1406267375-170667a.png')
        self.ufo_image = pg.transform.scale(self.ufo_image, (self.cell_size, self.cell_size))
        
        # Load the agent image
        self.agent_image = pg.image.load('star.jpg')
        self.agent_image = pg.transform.scale(self.agent_image, (self.cell_size, self.cell_size))
        
        # Load the goal image
        self.goal_image = pg.image.load('Massive_Black_Hole_at_the_Center_of_the_Milky_Way.jpg')
        self.goal_image = pg.transform.scale(self.goal_image, (self.cell_size, self.cell_size))

    def reset(self):
        self.state = np.array([0, 0])
        self.done = False
        self.reward = 0
        self.hell_hit_count = 0  # Reset the hell hit count
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0])**2 + 
            (self.state[1] - self.goal[1])**2
        )
        return self.state, self.info

    def add_hell_states(self, hell_state_coordinates):
        self.hell_states.append(np.array(hell_state_coordinates))

    def step(self, action):
        if action == 0 and self.state[0] > 0:  # Up
            self.state[0] -= 1
        elif action == 1 and self.state[0] < self.grid_size - 1:  # Down
            self.state[0] += 1
        elif action == 2 and self.state[1] < self.grid_size - 1:  # Right
            self.state[1] += 1
        elif action == 3 and self.state[1] > 0:  # Left
            self.state[1] -= 1

        if np.array_equal(self.state, self.goal):
            self.reward += 10
            self.done = True
        elif any(np.array_equal(self.state, each_hell) for each_hell in self.hell_states):
            self.hell_hit_count += 1
            if self.hell_hit_count == 1:
                self.reward += -3.33
            elif self.hell_hit_count == 2:
                self.reward += -3.33
            elif self.hell_hit_count == 3:
                self.reward = 0  # Set the reward to 0 after hitting the third hell state
                self.done = True
        else:
            self.reward += -0.05
            self.done = False

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0] - self.goal[0])**2 + 
            (self.state[1] - self.goal[1])**2
        )
        
        return self.state, self.reward, self.done, self.info

    def render(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()

        self.screen.blit(self.background_image, (0, 0))

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid = pg.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pg.draw.rect(self.screen, (200, 200, 200), grid, 1)

        for each_hell in self.hell_states:
            self.screen.blit(self.ufo_image, (each_hell[1] * self.cell_size, each_hell[0] * self.cell_size))

        self.screen.blit(self.goal_image, (self.goal[1] * self.cell_size, self.goal[0] * self.cell_size))
        self.screen.blit(self.agent_image, (self.state[1] * self.cell_size, self.state[0] * self.cell_size))

        # Display the reward
        font = pg.font.Font(None, 36)
        reward_text = font.render(f"Reward: {self.reward:.2f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (10, 10))

        pg.display.flip()

    def close(self):
        pg.quit()

    def show_welcome_message(self):
        font = pg.font.Font(None, 74)
        text = font.render("XoXo", True, (0, 128, 0))
        text_rect = text.get_rect(center=(self.cell_size * self.grid_size // 2, self.cell_size * self.grid_size // 2))

        self.screen.fill((255, 255, 255))
        self.screen.blit(text, text_rect)
        pg.display.flip()

        # Wait for user to press any key
        waiting = True
        while waiting:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()
                elif event.type == pg.KEYDOWN:
                    waiting = False

if __name__ == "__main__":
    my_env = ChidEnv()
    my_env.show_welcome_message()  # Show welcome message before starting the game

    # Prompt user to input the number of hell states
    num_hell_states = int(input("Enter the number of hell states: "))
    
    # Add hell states based on user input
    for _ in range(num_hell_states):
        x = int(input("Enter x coordinate for hell state: "))
        y = int(input("Enter y coordinate for hell state: "))
        my_env.add_hell_states([x, y])

    observation, info = my_env.reset()
    print(f"Initial state: {observation}, Info: {info}")

    for _ in range(15):
        # Choose your action [0, 1, 2, 3]
        action = int(input("Choose action: "))

        # Take the action in your environment
        new_state, reward, done, info = my_env.step(action)
        print(f"New state: {new_state}, Reward: {reward}, Done: {done}, Info: {info}")

        # Render the environment
        my_env.render()

        # Check for termination condition
        if done:
            my_env.render()  # Render the final state before game over
            my_env.close()
            print("I reached the goal!!" if reward > 0 else "Game over: Too many hell states hit!")
            break
    my_env.close()
