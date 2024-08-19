import numpy as np
import matplotlib.pyplot as plt
import random

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((5, 5), dtype=int)
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((5, 5), dtype=int)
        self.done = False
        self.winner = None
        return self.board

    def check_winner(self):
        for i in range(5):
            for j in range(3):
                if np.all(self.board[i, j:j+3] == 1) or np.all(self.board[j:j+3, i] == 1):
                    self.done = True
                    self.winner = 1
                    return
                if np.all(self.board[i, j:j+3] == -1) or np.all(self.board[j:j+3, i] == -1):
                    self.done = True
                    self.winner = -1
                    return
        
        for i in range(3):
            for j in range(3):
                if np.all([self.board[i+k, j+k] == 1 for k in range(3)]) or np.all([self.board[i+k, j+2-k] == 1 for k in range(3)]):
                    self.done = True
                    self.winner = 1
                    return
                if np.all([self.board[i+k, j+k] == -1 for k in range(3)]) or np.all([self.board[i+k, j+2-k] == -1 for k in range(3)]):
                    self.done = True
                    self.winner = -1
                    return
        
        if not np.any(self.board == 0):
            self.done = True
            self.winner = 0

    def step(self, action, player):
        if self.board[action] != 0 or self.done:
            raise ValueError("Invalid move or game is already finished")
        self.board[action] = player
        self.check_winner()
        return self.board, self.done, self.winner

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.get_q_value(state, action) for action in available_actions]
        max_q = max(q_values)
        return available_actions[q_values.index(max_q)]

    def learn(self, state, action, reward, next_state, available_actions):
        old_q_value = self.get_q_value(state, action)
        future_q_values = [self.get_q_value(next_state, a) for a in available_actions]
        max_future_q = max(future_q_values) if future_q_values else 0
        self.q_table[(state, action)] = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

def plot_board(ax, board):
    ax.clear()
    for i in range(1, 5):
        ax.plot([i, i], [0, 5], color="black", linewidth=3)
        ax.plot([0, 5], [i, i], color="black", linewidth=3)
    
    for i in range(5):
        for j in range(5):
            if board[i, j] == 1:
                ax.text(j + 0.5, 4.5 - i, 'X', va='center', ha='center', fontsize=36, color="red", fontweight='bold')
            elif board[i, j] == -1:
                ax.text(j + 0.5, 4.5 - i, 'O', va='center', ha='center', fontsize=36, color="blue", fontweight='bold')
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    plt.draw()

def visualize_game(env, agent, user_vs_ai=False):
    state = tuple(env.reset().flatten())
    done = False
    current_player = 1  # 1 for AI, -1 for Opponent or User
    
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.ion()  # Turn on interactive mode

    while not done:
        plot_board(ax, env.board)
        
        if current_player == 1:
            available_actions = [i for i in range(25) if state[i] == 0]
            action = agent.choose_action(state, available_actions)
            state, done, winner = env.step(divmod(action, 5), current_player)
        else:
            if user_vs_ai:
                plot_board(ax, env.board)
                plt.pause(0.1)
                while True:
                    try:
                        user_input = input("Enter your move (0-24): ").strip()
                        if not user_input.isdigit():
                            raise ValueError
                        user_action = int(user_input)
                        if user_action < 0 or user_action >= 25 or state[user_action] != 0:
                            raise ValueError
                        break
                    except ValueError:
                        print("Invalid input. Please enter a valid move (0-24) corresponding to an empty cell.")
                
                state, done, winner = env.step(divmod(user_action, 5), current_player)
            else:
                opponent_action = random.choice([i for i in range(25) if state[i] == 0])
                state, done, winner = env.step(divmod(opponent_action, 5), current_player)
        
        state = tuple(state.flatten())
        plt.pause(0.5)
        
        if done:
            break
        
        current_player *= -1
    
    plot_board(ax, env.board)
    plt.pause(1)
    
    if winner == 1:
        print("Agent Wins!")
    elif winner == -1:
        print("Opponent Wins!")
    else:
        print("It's a Tie!")

    plt.ioff()

# Example usage:
env = TicTacToeEnv()
agent = QLearningAgent()
visualize_game(env, agent, user_vs_ai=True)

