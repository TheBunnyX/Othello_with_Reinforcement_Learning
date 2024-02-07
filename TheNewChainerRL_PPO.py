import gym
import numpy as np
from gym import spaces
import random
import time
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import distributions
from playsound import playsound

np.random.seed(42)
random.seed(42)

BLACK = 1
WHITE = -1

class OthelloEnv(gym.Env):
    def __init__(self):
        self.board_size = 8
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board[3][3] = self.board[4][4] = -1
        self.board[3][4] = self.board[4][3] = 1
        self.current_player = 1
        self.done = False
        self.valid_moves = []

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board[3][3] = self.board[4][4] = -1
        self.board[3][4] = self.board[4][3] = 1
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
        self.current_player = -self.current_player

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

class PPOAgent(chainer.Chain):
    def __init__(self, obs_size, n_actions):
        super(PPOAgent, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(obs_size, 64)
            self.l2 = L.Linear(64, 64)
            self.pi = L.Linear(64, n_actions)
            self.v = L.Linear(64, 1)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        return self.pi(h), self.v(h)

def play_vs_random(env, episodes):
    player1_wins = 0
    player2_wins = 0
    draws = 0

    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False

        while not done:
            current_player = env.current_player
            valid_moves = env.get_valid_moves()

            if not valid_moves:
                no_valid_moves_count += 1
                if no_valid_moves_count >= 2:
                    break
            else:
                no_valid_moves_count = 0

            if current_player == BLACK:
                print("is = ",valid_moves)
                action = random.choice(valid_moves) if valid_moves else None
            else:
                action = random.choice(valid_moves) if valid_moves else None

            if action is not None:
                obs, reward, done, _ = env.step(action)

            env.render()

        winner = env.determine_winner()
        if winner == 1:  # Check for numerical values (1 for Player 1)
            player1_wins += 1
        elif winner == -1:  # Check for numerical values (-1 for Player 2)
            player2_wins += 1
        else:
            draws += 1
        env.render()

        if episode % 10 == 0 or episode == episodes:
            print(f"Episode: {episode}")
            print(f"Player 1 Wins: {player1_wins}, Player 2 Wins: {player2_wins}, Draw: {draws}")
        if episode == episodes:
            playsound('/path/note.wav')


    print("\nTotal wins for Player 1:", player1_wins)
    print("Total wins for Player 2:", player2_wins)
    print("Total for Draw:", draws)

if __name__ == "__main__":
    env = OthelloEnv()
    start_time = time.time()
    
    print("Playing Random Agent vs Random Agent...")
    play_vs_random(env, 1)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")