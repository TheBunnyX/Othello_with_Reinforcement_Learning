from tkinter import ttk
from tkinter import messagebox
import gym
import numpy as np
from gym import spaces
import random
import time
import tkinter as tk

#np.random.seed(42)
#random.seed(42)

BLACK = 1
WHITE = -1

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
        self.white_count = 0
        self.black_count = 0
        self.ep_count = 1
        self.set_board()

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
    '''
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
    '''
    #Check White black and Empty
    def white_score(self):
        white_count = self.white_count
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == -1:
                    white_count += 1
        return white_count

    def black_score(self):
        black_count = self.black_count
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] == 1:
                    black_count += 1
        return black_count

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
        #print("taking action...")
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
    
    def set_board(self):
        global root,canvas,white_label,black_label,current_player_label,episodes_label
        if(self.current_player == 1):
            current_player = "Black"
        elif(self.current_player == -1):
            current_player = "White"
        else:
            print(self.current_player)
        #set the main window
        root = tk.Tk()
        root.title("Othello")
        #Create canvas for the game board
        canvas = tk.Canvas(root, width=400, height=400, background="green")
        canvas.pack()
        #show labels for player white scores and black score 
        white_label = tk.Label(root, text=f"White: {self.white_count}")
        white_label.pack(side="left")
        black_label = tk.Label(root, text=f"Black: {self.black_count}")
        black_label.pack(side="right")
        #show label for current player
        current_player_label = tk.Label(root, text=f"Current Player : {current_player}", anchor="center")
        current_player_label.pack(fill="both", expand=True)
        #show label for episodes
        #episodes_label = tk.Label(root, text=f"Episodes : {self.ep_count}", anchor="center")
        #episodes_label.pack(fill="both", expand=True)

        #crate started place
        #update_board(,self.current_player)
        #Bind click event to canvas  #col and row
        #canvas.mainloop()
        #canvas.bind("<Button-1>", lambda event: self.step(event.x // 50,event.y // 50))

def update_board(env,current_player):
    #Get valid moves
        canvas.delete("all")
        #draw lines
        for i in range(8): ##ส้รางเส้นตาราง 8 เส้น
            canvas.create_line(i * 50, 0, i * 50, 400, fill="black", width=2)
            canvas.create_line(0, i * 50, 400, i * 50, fill="black", width=2)
        if(env.current_player == 1):
            current_player = "black"
        elif(env.current_player == -1):
            current_player = "white"
        else:
            print(env.current_player)
        #print("updating board...")
        valid_moves = env.get_valid_moves()
        for row in range(env.board_size):
            for col in range(env.board_size):
                x0, y0 = col * 50, row * 50
                x1, y1 = x0 + 50, y0 + 50
                if env.board[row][col] == -1:
                    canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="white", outline="black")
                elif env.board[row][col] == 1:
                    canvas.create_oval(x0 + 5, y0 + 5, x1 - 5, y1 - 5, fill="black", outline="black")
                elif (row, col) in valid_moves:
                    if current_player == "black" :
                        canvas.create_oval(x0 + 20, y0 + 20, x1 - 20, y1 - 20, fill = "black")
                    else :
                        canvas.create_oval(x0 + 20, y0 + 20, x1 - 20, y1 - 20, fill = "white")

        #print("done updating...")
        #canvas.mainloop()
        #canvas.bind("<Button-1>", lambda event: env.step(event.x // 50,event.y // 50))


def print_board(env):
    if(env.current_player == 1):
        current_player = "Black"
    elif(env.current_player == -1):
        current_player = "White"
    else:
        print(env.current_player)

    white_label.config(text=f"White: {env.white_score()}")
    black_label.config(text=f"Black: {env.black_score()}")
    current_player_label.config(text=f"Player: {current_player}")
    #episodes_label.config(text=f"Episodes : {env.ep_count}")
    canvas.update()
    #print("done printing...")
'''
def popup_winner(msg):
    print(msg)
    pop = tk.Tk()
    font = ("Verdana",16)
    pop.geometry("40x75")    
    pop.wm_title("WINNER")
    pop_label = ttk.Label(pop,test=msg,font=font)
    pop_label.pack(pady=10)

    P1 = ttk.Button(pop,text="GO GO! GO!",command= pop.destroy())
    P1.pack
    pop.mainloop()
'''


def play_vs_random(env, episodes):
    player1_wins = 0
    player2_wins = 0
    draws = 0
    #print("choose action...")
    for episode in range(1, episodes + 1):
        print("____Starting Episodes ",episode,"____")
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
            #
            if current_player == BLACK:
                action = random.choice(valid_moves) if valid_moves else None
            else:
                action = random.choice(valid_moves) if valid_moves else None
            #
            if action is not None:
                obs, reward, done, _ = env.step(action)

            time.sleep(0.1) 
            update_board(env,current_player)
            print_board(env)
            #print_board(env)

        winner = env.determine_winner()
        #print(winner)
        if winner == 1:  # Check for numerical values (1 for Player 1)
            player1_wins += 1
            messagebox.showinfo("___RESULT___","The winner is ...BLACK!")
            print("Round",episode,"The winner is BLACK")
        elif winner == -1:  # Check for numerical values (-1 for Player 2)
            player2_wins += 1
            messagebox.showinfo("___RESULT___","The winner is ...WHITE!")
            print("Round",episode,"The winner is WHITE")
        else:
            draws += 1
            messagebox.showinfo("___RESULT___"," Draw ;__; ")
            print("Round",episode,"is Draw :)")
        update_board(env,current_player)
        print_board(env)
        time.sleep(3)
        env.ep_count += 1
        

        if episode % 10 == 0 or episode == episodes:
            print(f"Episode: {episode}")
            print(f"Player 1 Wins: {player1_wins}, Player 2 Wins: {player2_wins}, Draw: {draws}")

    print("\nTotal wins for Player 1(Black):", player1_wins)
    print("Total wins for Player 2(White):", player2_wins)
    print("Total for Draw:", draws)


if __name__ == "__main__":
    env = OthelloEnv()
    start_time = time.time()

    print("Playing Random Agent vs Random Agent...")
    play_vs_random(env, 3)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")