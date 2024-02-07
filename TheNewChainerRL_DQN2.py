import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
from chainer import optimizers, Chain
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import Variable
import random
import time
import gym
from gym import spaces
from chainer import serializers
from playsound import playsound 
#from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import csv
import pandas as pd

import os

# Set the CUDA device to use GPU (For example, using GPU device 0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        flipped_count = 0
        if self.done:
            return self.board, 0, True, {}

        if action not in self.valid_moves:
            return self.board, -10, False, {}  # Invalid move penalized with -10 reward

        row = action // self.board_size
        col = action % self.board_size

        self.board[row][col] = self.current_player
        self.flip_discs(action)   # Pass only the action
        reward = flipped_count
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
            reward = 5 if winner == BLACK else -1 if winner == WHITE else -1
            return self.board, reward, True, {}

        # Update the valid moves for the current player
        self.valid_moves = self.get_valid_moves()
        return self.board, 0, False, {}

class DQN_Network(Chain):
    def __init__(self, input_dim, output_dim, l1_reg_coef=0.01, l2_reg_coef=0.01):
        super(DQN_Network, self).__init__(
            l1=L.Linear(input_dim, 64),
            l2=L.Linear(64, 32),
            l3=L.Linear(32, 18),
            l4=L.Linear(18, 6),
            out=L.Linear(6, output_dim)
        )
        print(input_dim, output_dim)


        self.l1_reg_coef = l1_reg_coef
        self.l2_reg_coef = l2_reg_coef
        #print(input_dim, output_dim)

        #Adding L1 and L2 regularization
        self.add_persistent('l1_reg', self.xp.zeros(1, dtype=self.xp.float32))
        self.add_persistent('l2_reg', self.xp.zeros(1, dtype=self.xp.float32))

    def __call__(self, x):
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))
        #h = F.sigmoid(self.l1(x))
        #h = F.sigmoid(self.l2(h))
        #h = F.tanh(self.l1(x))
        #h = F.tanh(self.l2(h))
        #return self.out(h)

        # Compute L1 and L2 regularization
        self.l1_reg = self.xp.absolute(self.l1.W.data).sum()
        self.l2_reg = 0.5 * self.xp.square(self.l1.W.data).sum()

        return self.out(h) + self.l1_reg_coef * self.l1_reg + self.l2_reg_coef * self.l2_reg


class DQN_Agent:
    def __init__(self, input_dim, output_dim, epsilon = 0.7, epsilon_decay=0.95, epsilon_min=0.01,
                 learning_rate=0.001, gamma=0.9, batch_size=32,replay_buffer_size=10000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size

        self.model = DQN_Network(input_dim, output_dim)
        self.target_model = DQN_Network(input_dim, output_dim)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        self.replay_buffer_size = replay_buffer_size
        self.replay_memory = []
        self.steps = 0
        self.target_update_freq = 5
   
    def save_model(self, file_path):
        serializers.save_npz(file_path, self.model)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        loaded_model = DQN_Network(self.input_dim, self.output_dim)
        serializers.load_npz(file_path, loaded_model)
        self.model = loaded_model
        print(f"Model loaded from {file_path}")

    def checkpoint_model(self, file_path):
        loaded_model = DQN_Network(self.input_dim, self.output_dim)
        serializers.load_npz(file_path, loaded_model)
        self.model = loaded_model
        print(f"Model loaded from {file_path}")

    def loss_function(self, y_pred, y_true):
        # Calculate the loss using Huber loss function (you can modify this based on your task)
        loss = F.mean(F.huber_loss(y_pred, y_true, delta=1.0)) #loss = F.mean_squared_error(y_pred, y_true) 
        return loss

    def update_target_model(self):
        self.target_model = self.model.copy()

    def remember(self, state, action, reward, next_state, done):
        # Store experiences in the replay buffer
        if len(self.replay_memory) < self.replay_buffer_size:
            self.replay_memory.append((state, action, reward, next_state, done))
        else:
            #Remove the oldest experience if the buffer is full
            self.replay_memory.pop(0)
            self.replay_memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)
        else:
            q_values = self.model(np.array([state]).astype(np.float32))
            q_values = q_values.data[0]
            masked_q_values = [q_values[i] if i in valid_moves else -np.inf for i in range(self.output_dim)]
           
            q_values_df = pd.DataFrame({'Q-values': masked_q_values})
            print(q_values_df)  # print DataFrame

            return np.argmax(masked_q_values)

    def replay(self):
        if len(self.replay_memory) < self.batch_size:
            return

        batch = random.sample(self.replay_memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = self.model(np.array([state]).astype(np.float32)).data[0]
            if done:
                target[action] = reward
            else:
                Q_future = max(self.target_model(np.array([next_state]).astype(np.float32)).data[0])
                target[action] = reward + self.gamma * Q_future
            states.append(state)
            targets.append(target)
        
        # Update the optimizer with the loss function
        self.optimizer.update(self.loss_function, np.array(states).astype(np.float32), np.array(targets).astype(np.float32))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.steps % self.target_update_freq == 0:
            self.update_target_model()

        self.steps += 1

class OthelloChainerRLWrapper(gym.Wrapper):
    def __init__(self, env, agent):
        super(OthelloChainerRLWrapper, self).__init__(env)
        self.agent = agent

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def train_vs_dqn(self, episodes):
        player1_wins = 0
        player2_wins = 0
        draws = 0      

        player1_rewards = []
        player2_rewards = []

        player1_ = 0
        player2_ = 0
        draw = 0

        count_player1_wins = []
        count_player2_wins = []
        count_draws = []

        checkpoint = True
        checkpoint_frequency = 125  # Save checkpoint every 10 episodes

        for episode in range(1, episodes + 1):
            obs = self.reset()
            done = False

            episode_reward_player1 = 0
            episode_reward_player2 = 0
            
            while not done:
                valid_moves = self.env.get_valid_moves()

                if not valid_moves:
                    break

                if self.env.current_player == BLACK:  # DQN Agent's turn
                    if obs is not None:  # Check if obs is not None
                        action = self.agent.choose_action(obs.flatten(), valid_moves)
                    else:
                        #print("action with random move")
                        action = random.choice(valid_moves)
                else:  # Random Agent's turn
                    action = random.choice(valid_moves)

                obs, reward, done, _ = self.step(action)

                if self.env.current_player == BLACK:
                    episode_reward_player1 +=1
                else:  # Random Agent's turn
                    episode_reward_player2 +=1
                if obs is not None:  # Check if obs is not None after the step
                    self.agent.remember(obs.flatten(), action, reward, obs.flatten(), done)
                    self.agent.replay()

                winner = env.determine_winner()
                if winner == 1:  # Check for values (1 for Player 1)
                    player1_+1
                elif winner == -1:  # Check for values (-1 for Player 2)
                    player2_+1
                else :
                    draw+1
                    
                winner = env.determine_winner()
                if winner == 1:  # Check for nvalues (1 for Player 1)
                    player1_ += 1
                elif winner == -1:  # Check for values (-1 for Player 2)
                    player2_ += 1
                else:
                    draw += 1  # Increment draw count

            self.agent.epsilon *= self.agent.epsilon_decay    

            #env.render()
                
            winner = env.determine_winner()
            if winner == 1:
                player1_wins += 1
            elif winner == -1:
                player2_wins += 1
            else:
                draws += 1

            count_player1_wins.append(player1_wins)
            count_player2_wins.append(player2_wins)
            count_draws.append(draws)

            player1_rewards.append(episode_reward_player1/episodes + 1)
            player2_rewards.append(episode_reward_player2/episodes + 1)


            if episode % 10 == 0:
                print(f"Episode: {episode}")
                print(f"Player 1 Wins: {player1_wins}, Player 2 Wins: {player2_wins}, Draw: {draws}")

                file_path = 'C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/episode_results.csv'

                # Write to CSV
                with open(file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if episode == 10:
                        writer.writerow(['Episode', 'Player 1 Wins', 'Player 2 Wins', 'Draws'])
                    writer.writerow([episode, player1_wins, player2_wins, draws])

            if episode % checkpoint_frequency == 0 and checkpoint == True:
                checkpoint_path = f'C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/checkpoint/checkpoint_{episode}.npz'
                dqn_agent.save_model(checkpoint_path)
                print(f"Checkpoint saved at episode {episode} to {checkpoint_path}")
           
            if episode == episodes:
                playsound('C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/finishsound.mp3')

        print("\nTotal wins for Player 1:", player1_wins)
        print("Total wins for Player 2:", player2_wins)
        print("Total for Draw:", draws)
        
        #writer.close()

        # Plotting
        plt.figure(figsize=(10, 10))

        # Plotting wins, losses, and draws per episode
        plt.subplot(2, 1, 1)
        plt.plot(range(1, episodes + 1), np.cumsum(count_player1_wins) , label='Player 1 Wins')
        plt.plot(range(1, episodes + 1), np.cumsum(count_player2_wins) , label='Player 2 Wins')
        plt.plot(range(1, episodes + 1),  np.cumsum(count_draws) , label='Draws')
        plt.xlabel('Episodes')
        plt.ylabel('winrate')
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
        plt.savefig('C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/PictureRandom/RewardperEpisodeforPlayersOfDQN.png')
        #plt.show()

    def play_vs_dqn(self, episodes):
        player1_wins = 0
        player2_wins = 0
        draws = 0        

        player1_rewards = []
        player2_rewards = []

        count_player1_wins = []
        count_player2_wins = []
        count_draws = []
        episode_reward_player1 = 0
        episode_reward_player2 = 0

        for episode in range(1, episodes + 1):
            obs = self.reset()
            done = False
    
            while not done:
                valid_moves = self.env.get_valid_moves()

                if not valid_moves:
                    break

                if self.env.current_player == BLACK:  # DQN Agent's turn
                    if obs is not None:  # Check if obs is not None
                        action = self.agent.choose_action(obs.flatten(), valid_moves)
                    else:
                        #print("action with random move")
                        action = random.choice(valid_moves)
                else:  # Random Agent's turn
                    action = random.choice(valid_moves)

                obs, reward, done, _ = self.step(action)
                if self.env.current_player == BLACK:
                    episode_reward_player1 += reward
                else:  # Random Agent's turn
                    episode_reward_player2 += reward

                if obs is not None:  # Check if obs is not None after the step
                    self.agent.remember(obs.flatten(), action, reward, obs.flatten(), done)
                    self.agent.replay()

            env.render()
                
            winner = env.determine_winner()
            if winner == 1:  # Check for numerical values (1 for Player 1)
                player1_wins += 1
            elif winner == -1:  # Check for numerical values (-1 for Player 2)

                player2_wins += 1
            else:
                draws += 1

            count_player1_wins.append(player1_wins)
            count_player2_wins.append(player2_wins)
            count_draws.append(draws)

            player1_rewards.append(episode_reward_player1/episodes + 1)
            player2_rewards.append(episode_reward_player2/episodes + 1)

            if episode % 10 == 0 :
                print(f"Episode: {episode}")
                print(f"Player 1 Wins: {player1_wins}, Player 2 Wins: {player2_wins}, Draw: {draws}")
           
            if episode == episodes:
                playsound('C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/finishsound.mp3')

        print("\nTotal wins for Player 1:", player1_wins)
        print("Total wins for Player 2:", player2_wins)
        print("Total for Draw:", draws)
        
        #writer.close()

        # Plotting
        plt.figure(figsize=(10, 10))

        # Plotting wins, losses, and draws per episode
        plt.subplot(2, 1, 1)
        plt.plot(range(1, episodes + 1), np.cumsum(count_player1_wins), label='Player 1 Wins')
        plt.plot(range(1, episodes + 1), np.cumsum(count_player2_wins), label='Player 2 Wins')
        plt.plot(range(1, episodes + 1), np.cumsum(count_draws), label='Draws')
        plt.xlabel('Episodes')
        plt.ylabel('Count')
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
        plt.savefig('C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/PictureRandom/AfterRewardperEpisodeforPlayersOfOfDQN.png')
        #plt.show()
        

if __name__ == "__main__":
    env = OthelloEnv()
    input_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    output_dim = env.action_space.n

    dqn_agent = DQN_Agent(input_dim, output_dim)
    othello_chainer_rl = OthelloChainerRLWrapper(env, dqn_agent)
    start_time1 = time.time()

    print("Training DQN Agent vs Random Agent...")
    othello_chainer_rl.train_vs_dqn(10)  # Play 100 episodes
    dqn_agent.save_model('C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/dqn_model.npz')

    end_time1 = time.time()

    elapsed_time_train = end_time1 - start_time1

    start_time2 = time.time()
    dqn_agent.load_model('C:/Users/MongkolChut/Downloads/New folder/GymRandom/Finished/checkpoint/checkpoint_10.npz')

    othello_chainer_rl.play_vs_dqn(10)
    end_time2 = time.time()
   
    elapsed_time_test = end_time2 - start_time2

    print(f"Elapsed time: {elapsed_time_train} seconds")
    print(f"Elapsed time: {elapsed_time_test} seconds")

    #tensorboard --logdir=runs