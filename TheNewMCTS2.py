import gym
import numpy as np
from gym import spaces
import random
import time
import math
from collections import defaultdict
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

BLACK = 1
WHITE = 2

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.wins = 0
        self.visits = 0
        self.children = {}
        self.untried_moves = state.get_valid_moves()
        self.parent = parent

    def select_child(self):
        C = 1.4  # UCT constant, you can adjust this value
        if self.children:
            return max(
                self.children.items(),
                key=lambda c: c[1].wins / c[1].visits + C * math.sqrt(2 * math.log(self.visits) / c[1].visits),)[0]
        else:
            return None

class OthelloEnv(gym.Env):
    def __init__(self):
        self.board_size = 8
        self.action_space = spaces.Discrete(self.board_size ** 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.board_size, self.board_size), dtype=np.int8)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.board[3][3] = self.board[4][4] = WHITE
        self.board[3][4] = self.board[4][3] = BLACK
        self.current_player = BLACK
        self.done = False
        self.valid_moves = []

    def clone(self):
        cloned_env = OthelloEnv()
        cloned_env.board_size = self.board_size
        cloned_env.action_space = self.action_space
        cloned_env.observation_space = self.observation_space
        cloned_env.board = np.copy(self.board)
        cloned_env.current_player = self.current_player
        cloned_env.done = self.done
        cloned_env.valid_moves = self.valid_moves.copy()
        return cloned_env

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

    def simulate_random_playout(self):
        current_player = self.current_player
        temp_board = np.copy(self.board)

        while not self.done:
            valid_moves = self.get_valid_moves()

            if not valid_moves:
                self.current_player = -self.current_player
                valid_moves = self.get_valid_moves()

                if not valid_moves:
                    self.done = True
                    break

            random_action = random.choice(valid_moves)
            self.step(random_action)

        winner = self.determine_winner()
        if winner == BLACK:
            return 1
        elif winner == WHITE:
            return -1
        else:
            return 0

    def run_mcts(env, num_iterations):
        root = Node(env.clone())

        for _ in range(num_iterations):
            node = root
            state = node.state.clone()

            # Selection
            while not node.untried_moves and node.children:
                action = node.select_child()
                state.step(action)
                node = node.children[action]

            # Expansion
            if node.untried_moves:
                action = random.choice(node.untried_moves)
                state.step(action)
                new_node = Node(state)
                node.children[action] = new_node
                node = new_node

            # Simulation
            state.simulate_random_playout()

            # Backpropagation
            while node:
                node.visits += 1
                node.wins += state.determine_winner() == BLACK  #Counting wins for BLACK player
                node = node.parent

        return max(root.children.items(), key=lambda c: c[1].visits)[0]

def play_vs_mcts(env, episodes, iterations):
    player1_wins = 0
    player2_wins = 0
    draws = 0

    player1_rewards = []
    player2_rewards = []

    count_player1_wins = []
    count_player2_wins = []
    count_draws = []


    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False

        episode_reward_player1 = 0
        episode_reward_player2 = 0
        
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
                action = env.run_mcts(iterations)
            else:
                action = random.choice(valid_moves) if valid_moves else None

            if action is not None:
                obs, reward, done, _ = env.step(action)
                if current_player == BLACK:
                    episode_reward_player1 += reward
                else:  # Random Agent's turn
                    episode_reward_player2 += reward

            #env.render()
        winner = env.determine_winner()
        if winner == 1:
            player1_wins += 1
        elif winner == -1:
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

    # Plotting
    plt.figure(figsize=(10, 10))

    # Plotting wins, losses, and draws per episode
    plt.subplot(2, 1, 1)
    plt.plot(range(1, episodes + 1), np.cumsum(count_player1_wins), label='Player 1 Wins')
    plt.plot(range(1, episodes + 1), np.cumsum(count_player2_wins), label='Player 2 Wins')
    plt.plot(range(1, episodes + 1), np.cumsum(count_draws), label='Draws')
    plt.xlabel('Episodes')
    plt.ylabel('Winrate')
    plt.title('Wins, Losses, and Draws per Episode')
    plt.legend()
       
        #plt.savefig('C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/WinsLossesandDrawsperEpisodeOfRandom.png')

        # Plotting cumulative rewards per episode for each player
    plt.subplot(2, 1, 2)
    plt.plot(range(1, episodes + 1), np.cumsum(player1_rewards), label='Player 1 Rewards')
    plt.plot(range(1, episodes + 1), np.cumsum(player2_rewards), label='Player 2 Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('Reward per Episode for Players')
    plt.legend()

    plt.tight_layout()
    plt.savefig('C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/PictureRandom/RewardperEpisodeforPlayersOfOfMCTS.png')

if __name__ == "__main__":
    env = OthelloEnv()
    start_time = time.time()

    print("Random Agent vs Playing MCTS Agent...")
    play_vs_mcts(env, 1, iterations = 2)  

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")