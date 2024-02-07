import gym
import numpy as np
from gym import spaces
import random
import time

np.random.seed(42)
random.seed(42)

WHITE = -1
BLACK = 1


class OthelloEnv(gym.Env):
    def __init__(self):
        self.board_size = 8
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board[3][3] = self.board[4][4] = -1
        self.board[3][4] = self.board[4][3] = 1
        self.players = [1, -1]
        self.current_player = 1
        self.done = False
        self.valid_moves = []

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board[3][3] = self.board[4][4] = -1
        self.board[3][4] = self.board[4][3] = 1
        self.players = [1, -1]
        self.current_player = 1
        self.done = False
        self.valid_moves = []

    def get_valid_moves(self):
        valid_moves = []

        player = self.current_player
        opponent = -player

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                        dx, dy = direction
                        x, y = i + dx, j + dy

                        # Move must lead to a valid capture in at least one direction
                        if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == opponent:
                            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == opponent:
                                x += dx
                                y += dy

                            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == player:
                                valid_moves.append((i, j))
                                break

        self.valid_moves = valid_moves
        return valid_moves

    def is_valid_move(self, row, col, player):
        if self.board[row][col] != 0:
            return False

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        valid_move = False

        for d in directions:
            dx, dy = d
            x, y = row + dx, col + dy
            temp_flip = False

            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == -player:
                x += dx
                y += dy
                temp_flip = True

            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == player and temp_flip:
                valid_move = True
                break

        return valid_move

    def flip_discs(self, row, col, player):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]

        for d in directions:
            dx, dy = d
            x, y = row + dx, col + dy
            discs_to_flip = []

            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == -player:
                discs_to_flip.append((x, y))
                x += dx
                y += dy

            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == player:
                for disc in discs_to_flip:
                    self.board[disc[0]][disc[1]] = player

    def has_legal_moves(self):
        return len(self.get_valid_moves()) > 0

    def render(self, mode='human'):
        print(self.current_player)
        if self.current_player == -1:
            current_player_symbol = "W"
        else:
            current_player_symbol = "B"
        current_player_color = "\033[1;32m"  # Green for current player's discs
        player1_color = "\033[1;34m"        # White for Player 1's discs
        player2_color = "\033[1;37m"        # Black for Player 2's discs
        valid_move_color = "\033[1;31m"     # Cyan for valid moves

        print(f"Current Player: {current_player_color}{current_player_symbol}\033[0m")

        for i, row in enumerate(self.board):
            rendered_row = []
            for j, cell in enumerate(row):
                if (i * self.board_size) + j in self.valid_moves:
                    rendered_row.append(f"{valid_move_color}*")
                else:
                    if cell == 1:
                        rendered_row.append(f"{player1_color}B")
                    elif cell == -1:
                        rendered_row.append(f"{player2_color}W")
                    else:
                        rendered_row.append(f"{current_player_color}-")
            print(" ".join(rendered_row) + "\033[0m")
        print("\n")

    def determine_winner(self):
        black_count = np.count_nonzero(self.board == 1)  # Change BLACK to 1
        white_count = np.count_nonzero(self.board == -1)  # Change WHITE to -1

        if black_count > white_count:
            return 1  # Return 1 for BLACK
        elif white_count > black_count:
            return -1  # Return -1 for WHITE
        else:
            return 0  # Return 0 for Draw

    def step(self, action):
        if self.done:
            return self.board, 0, True, {}

        if action not in self.valid_moves:
            return self.board, -10, False, {}  # Invalid move penalized with -10 reward

        row, col = action
        self.board[row][col] = self.current_player
        self.flip_discs(row, col, self.current_player)

        # Switch to the opposite player's turn
        #self.current_player = -self.current_player
        self.current_player = self.players[(self.players.index(self.current_player) + 1) % len(self.players)]

        if not self.has_legal_moves():
            # If no legal moves for the current player, switch to the other player's turn
            self.current_player = -self.current_player

            if not self.has_legal_moves():  # If neither player has legal moves
                self.done = True

        if np.count_nonzero(self.board == 0) == 0 or not self.has_legal_moves():
            # Game ends if the board is full or neither player has legal moves
            self.done = True

        if self.done:
            # Determine the winner and provide rewards accordingly
            winner = self.determine_winner()
            reward = 1 if winner == BLACK else -1 if winner == WHITE else 0
            return self.board, reward, True, {}

        # Update the valid moves for the current player
        self.valid_moves = self.get_valid_moves()
        return self.board, 0, False, {}

    def observe(self, player):
        # Define observation for each player
        if player == 1:
            return np.where(self.board == 1, 1, 0)
        elif player == -1:
            return np.where(self.board == -1, 1, 0)
        else:
            return np.zeros((self.board_size, self.board_size), dtype=np.int8)

import ray
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class OthelloMultiAgentEnv(MultiAgentEnv):
    def __init__(self):
        self.agents = {}
        self.env = OthelloEnv()

    def reset(self):
        obs = self.env.reset()
        self.agents = {
            str(player): player for player in [1, -1]
        }
        return {str(player): obs for player in [1, -1]}

    def step(self, actions):
        rewards = {}
        dones = {}
        infos = {}

        for player_str, action in actions.items():
            player = int(player_str)
            obs, reward, done, info = self.env.step(action)
            rewards[str(player)] = reward
            dones[str(player)] = done
            infos[str(player)] = info

        obs = {str(player): self.env.observe(player) for player in [1, W-1]}
        return obs, rewards, dones, infos


# Initialize Ray
ray.init()

# Define DQN trainers for each player
trainer1 = DQNConfig(env="OthelloMultiAgentEnv")  # Trainer for Player 1 (Black)
trainer2 = DQNConfig(env="OthelloMultiAgentEnv")  # Trainer for Player 2 (White)

# Training loop
for i in range(10):  # Train for 10 iterations
    done = False
    obs = env.reset()

    while not done:
        if i % 2 == 0:  # Player 1's turn
            action = trainer1.compute_action(obs["1"])  # Select action using DQN trainer
        else:  # Player 2's turn
            action = env.action_space.sample()  # Select random action

        next_obs, reward, done, _ = env.step({str(1): action, str(-1): action})

        if i % 2 == 0:  # Store experience for Player 1
            trainer1.train_batch(env.observe(1), action, reward, next_obs["1"], done)

        obs = next_obs

# Shutdown Ray
ray.shutdown()