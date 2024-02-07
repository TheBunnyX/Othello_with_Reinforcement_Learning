import gymnasium as gym
import numpy as np
from gym import spaces
import random
import time

#np.random.seed(42)
#random.seed(42)

BLACK = 1
WHITE = -1

class OthelloEnv(gym.Env):
    def __init__(self, num_envs=1):  # Update the constructor to accept the num_envs parameter
        self.num_envs = num_envs  # Add the num_envs attribute
        self.board_size = 8
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board[3][3] = self.board[4][4] = 1
        self.board[3][4] = self.board[4][3] = -1
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

        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == 0:
                    for direction in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                        dx, dy = direction
                        x, y = row + dx, col + dy
                        valid = False

                        while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == opponent:
                            x += dx
                            y += dy
                            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == player:
                                valid = True
                                break

                        if valid:
                            valid_moves.append(row * self.board_size + col)
                            break

        self.valid_moves = valid_moves
        return valid_moves

    def is_valid_move(self, action, player):
        row = action // self.board_size
        col = action % self.board_size

        if self.board[row][col] != 0:
            return False

        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            dx, dy = direction
            x, y = row + dx, col + dy
            temp_flip = False

            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == -player:
                x += dx
                y += dy
                temp_flip = True

            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == player and temp_flip:
                return True

        return False

    def flip_discs(self, action):
        row = action // self.board_size
        col = action % self.board_size

        player = self.current_player
        opponent = -player

        for direction in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            dx, dy = direction
            x, y = row + dx, col + dy
            discs_to_flip = []

            while 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == opponent:
                discs_to_flip.append((x, y))
                x += dx
                y += dy

            if 0 <= x < self.board_size and 0 <= y < self.board_size and self.board[x][y] == player:
                for disc in discs_to_flip:
                    self.board[disc[0]][disc[1]] = player

    def has_legal_moves(self):
        self.valid_moves = self.get_valid_moves()
        return len(self.valid_moves) > 0

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

        row = action // self.board_size
        col = action % self.board_size

        self.board[row][col] = self.current_player
        self.flip_discs(action)  # Pass only the action

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
                #print("is = ",valid_moves)
                if valid_moves:
                    action = random.choice(valid_moves)
                else:
                    action = None
            else:
                if valid_moves:
                    action = random.choice(valid_moves)
                else:
                    action = None

            if action is not None:
                obs, reward, done, _ = env.step(action)

            #env.render()

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

    print("\nTotal wins for Player 1:", player1_wins)
    print("Total wins for Player 2:", player2_wins)
    print("Total for Draw:", draws)



if __name__ == "__main__":
    env = OthelloEnv()
    start_time = time.time()
    
    print("Playing Random Agent vs Random Agent...")
    play_vs_random(env, 10)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")